#!/usr/bin/env python3
"""Export a ROCm artifact with quantized projections (RDNA / gfx11 WMMA kernels).

  source env.sh   # TVM built with configs/tvm/rocm.cmake
  # int8 MLP, fp16 attention, int8 logits (no calibration needed)
  python scripts/export_rocm_quant.py --preset int8-mlp
  # int4 MLP from an AMD Quark GPTQ export (see scripts/quantize_quark.py)
  python scripts/export_rocm_quant.py --preset int4-mlp --quark quant/fdllm_q4gptq_mlp
  # any mix: f16 (model dtype) | w8 | w4 per projection type
  python scripts/export_rocm_quant.py --precision qkv=f16,o=f16,gate_up=w4,down=w8 --quark ...

Weights:
- w4 projections come from the Quark int4 export.
- All other tensors come from the Quark export when one is given, otherwise from the HF checkpoint.
  The Quark export keeps every tensor it did not quantize, including AWQ-rescaled norms.
- w8 projections and the int8 logits are quantized here: group 128, round-to-nearest, from those fp16 weights.

Everything else (native model, opt_fast_dllm_llm_native pipeline, in-place KV cache manifest, smoke
run) is as in scripts/export_rocm.py.
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import tvm
from tvm import relax

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fast_dllm_relax.common import DEFAULT_MODEL_NAME, DEFAULT_MODEL_REVISION  # noqa: E402
from fast_dllm_relax.export_utils import load_hf_weights, make_relax_config, write_kv_cache_manifest  # noqa: E402
from fast_dllm_relax.model.metal import FastDLLMRelaxForCausalLMNative  # noqa: E402
from fast_dllm_relax.model.rocm_linear import (  # noqa: E402
    PRECISIONS,
    PRESETS,
    PROJECTIONS,
    add_w8_lm_head,
    quantize_rows_int8_g128,
    swap_projections,
)
from fast_dllm_relax.targets.rocm import (  # noqa: E402
    ROCM_HOST_TARGET_CONFIG,
    has_rdna_wmma,
    rocm_arch,
    rocm_target,
    rocm_target_config,
)

# HF checkpoint Linear names behind each fused projection.
PARTS = {"qkv": ["q_proj", "k_proj", "v_proj"], "o": ["o_proj"], "gate_up": ["gate_proj", "up_proj"],
         "down": ["down_proj"]}


def parse_precision(text: str) -> dict:
    spec = dict(kv.split("=", 1) for kv in text.split(","))
    if set(spec) != set(PROJECTIONS) or not set(spec.values()) <= set(PRECISIONS):
        raise SystemExit(f"--precision must set each of {PROJECTIONS} to one of {PRECISIONS}: {text!r}")
    return spec


def load_weights(args, need_int4: bool):
    """(hf_config, fp-tensor state dict, int4 dict)."""
    from transformers import AutoConfig  # noqa: PLC0415

    if args.quark:
        from fast_dllm_relax.quark_int4 import load_quark_int4  # noqa: PLC0415

        hf_config = AutoConfig.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
        int4, sd = load_quark_int4(args.quark)
        return hf_config, sd, int4
    if need_int4:
        raise SystemExit("w4 projections need --quark <AMD Quark int4 export>")
    hf_config, sd = load_hf_weights(args.model, args.revision)
    return hf_config, sd, {}


def map_params(cfg, spec, sd, int4, named, lm_head_w8: bool) -> dict:
    f16 = lambda t: t.detach().to(torch.float16).numpy()  # noqa: E731
    mapped = {name: f16(sd[name]) for name in named if name in sd}
    for i in range(cfg.num_hidden_layers):
        for proj, parts in PARTS.items():
            owner = f"model.layers.{i}." + ("self_attn" if proj in ("qkv", "o") else "mlp")
            src, dst = [f"{owner}.{p}" for p in parts], f"{owner}.{proj}_proj"
            if spec[proj] == "w4":
                missing = [s for s in src if s not in int4]
                if missing:
                    raise SystemExit(f"{missing[0]} is not int4 in the Quark export")
                mapped[f"{dst}.weight_packed"] = np.concatenate([int4[s][0] for s in src], axis=0)
                mapped[f"{dst}.weight_scales"] = np.concatenate([int4[s][1] for s in src], axis=0)
                continue
            missing = [s for s in src if f"{s}.weight" not in sd]
            if missing:
                raise SystemExit(f"{missing[0]}.weight is quantized in the Quark export; it cannot be {spec[proj]}")
            if spec[proj] == "f16":
                mapped[f"{dst}.weight"] = np.concatenate([f16(sd[f"{s}.weight"]) for s in src], axis=0)
            else:
                w = torch.cat([sd[f"{s}.weight"].to(torch.float16) for s in src], dim=0)
                mapped[f"{dst}.weight_packed"], mapped[f"{dst}.weight_scales"] = quantize_rows_int8_g128(w)
        attn = f"model.layers.{i}.self_attn"
        mapped[f"{attn}.qkv_proj.bias"] = np.concatenate([f16(sd[f"{attn}.{p}_proj.bias"]) for p in "qkv"], axis=0)
    if lm_head_w8:
        mapped["lm_head_w8.weight_packed"], mapped["lm_head_w8.weight_scales"] = quantize_rows_int8_g128(
            sd["model.embed_tokens.weight"].to(torch.float16))
    missing = sorted(set(named) - set(mapped))
    if missing:
        raise RuntimeError(f"unmapped params: {missing[:8]}")
    for name, p in named.items():
        if tuple(mapped[name].shape) != tuple(int(d) for d in p.shape) or str(mapped[name].dtype) != p.dtype:
            raise RuntimeError(f"{name}: {mapped[name].shape}/{mapped[name].dtype} vs {p.shape}/{p.dtype}")
    return mapped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--preset", choices=sorted(PRESETS), help="int8-mlp (recommended) | int4-mlp | fp16")
    sel.add_argument("--precision", help="qkv=..,o=..,gate_up=..,down=.. with f16 | w8 | w4")
    ap.add_argument("--quark", default=None, help="AMD Quark int4 g128 export dir (required for w4)")
    ap.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HF model (config; weights when no --quark)")
    ap.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    ap.add_argument("--arch", default=None, help="AMDGPU arch (default: FAST_DLLM_ROCM_ARCH or gfx1151)")
    ap.add_argument("--no-lm-head-w8", action="store_true", help="keep the fp16 tied logits projection")
    ap.add_argument("--suffix", default=None, help="artifact dir suffix (default: _<preset> or _mixed)")
    ap.add_argument("--no-smoke", action="store_true", help="skip the on-device smoke run")
    args = ap.parse_args()

    spec = dict(PRESETS[args.preset]) if args.preset else parse_precision(args.precision)
    arch, dtype = args.arch or rocm_arch(), "float16"
    if not has_rdna_wmma(arch):
        raise SystemExit(f"quantized projections need gfx11 WMMA; {arch} is not supported")
    suffix = args.suffix if args.suffix is not None else f"_{args.preset or 'mixed'}"
    artifact_dir = REPO_ROOT / "artifacts" / f"fast_dllm_relax_fp16_rocm_{arch}{suffix}"

    hf_config, sd, int4 = load_weights(args, "w4" in spec.values())
    cfg = make_relax_config(hf_config)
    model = FastDLLMRelaxForCausalLMNative(cfg, dtype=dtype, fused_qkv=True, fused_gate_up=True)
    swap_projections(model, cfg, dtype, spec)
    if not args.no_lm_head_w8:
        add_w8_lm_head(model, cfg, dtype)
    mod, named = model.export_tvm(spec=model.get_default_spec())
    named = dict(named)
    mapped = map_params(cfg, spec, sd, int4, named, not args.no_lm_head_w8)

    print(f"Compiling {spec} for {arch} ...")
    t0 = time.perf_counter()
    ex = tvm.compile(mod, rocm_target(arch),
                     relax_pipeline=relax.get_pipeline("opt_fast_dllm_llm_native", gemm_acc="float32"))
    compile_s = time.perf_counter() - t0
    artifact_dir.mkdir(parents=True, exist_ok=True)
    so_path = artifact_dir / "fast_dllm_relax_nn.so"
    ex.export_library(str(so_path))
    np.savez(artifact_dir / "fast_dllm_relax_nn_params.npz", *[mapped[n] for n in named])
    (artifact_dir / "fast_dllm_relax_nn_param_names.txt").write_text("\n".join(named) + "\n", encoding="utf-8")
    (artifact_dir / "fast_dllm_relax_nn_dtype.txt").write_text(dtype + "\n", encoding="utf-8")
    write_kv_cache_manifest(artifact_dir, cfg, dtype)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "tvm_version": tvm.__version__,
        "target_device": rocm_target_config(arch), "target_host": ROCM_HOST_TARGET_CONFIG,
        "pipeline": "opt_fast_dllm_llm_native", "wmma": True, "precision": spec,
        "lm_head_w8": not args.no_lm_head_w8, "max_total_len": None,
        "weights": {"model": args.model, "revision": args.revision, "quark_int4": args.quark},
        "build_host": {"system": platform.system(), "machine": platform.machine()},
        "param_bytes": int(sum(mapped[n].nbytes for n in named)), "compile_seconds": round(compile_s, 1),
    }
    (artifact_dir / "fast_dllm_relax_nn_rocm_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  compiled in {compile_s:.1f}s; params {manifest['param_bytes'] / 1e9:.2f} GB -> {artifact_dir}")

    if not args.no_smoke:
        from export_rocm import smoke_run  # noqa: PLC0415

        print("\nVerifying artifact on rocm:0 ...")
        smoke_run(artifact_dir, so_path, cfg, named, dtype, tvm.rocm(0))
    print(f"\n  python -m fast_dllm_relax.cli --backend relax --use-block-cache --device rocm:0 "
          f"--artifact-dir {artifact_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
