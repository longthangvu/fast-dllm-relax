#!/usr/bin/env python3
"""Compile the Fast-dLLM v2 Relax artifact for an AMD ROCm GPU (built and run on the same machine).

Builds `FastDLLMRelaxForCausalLMNative` (`fast_dllm_relax/model/metal.py`: in-place KV cache written
and attended in-graph, no Python callbacks, plus static-shape variants and on-device confidence)
with the target-agnostic `opt_fast_dllm_llm_native` pipeline (dlight only, no vendor libraries) for
`rocm -mcpu=<arch>` (default gfx1151, Strix Halo; override with --arch or FAST_DLLM_ROCM_ARCH).
RDNA GPUs run wave32, so the target sets thread_warp_size=32 (TVM's rocm default is 64).
Writes:

  artifacts/fast_dllm_relax_<fp16|bf16>_rocm_<arch>[<suffix>]/
      fast_dllm_relax_nn.so               host code + AMDGPU code object (hsaco)
      fast_dllm_relax_nn_params.npz / _param_names.txt / _dtype.txt
      fast_dllm_relax_nn_rocm_manifest.json
      fast_dllm_relax_nn_kv_cache.json    KV-cache contract (export_utils.KV_CACHE_CONTRACT)

Then loads the artifact on rocm:0 and smoke-runs the in-place KV-cache functions (each call may
write only its own cache rows).

Usage (repo root, TVM env from `bash scripts/setup_tvm.sh rocm`):
    python scripts/export_rocm.py                     # fp16, gfx1151
    python scripts/export_rocm.py --dtype bfloat16
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

from fast_dllm_relax.common import DEFAULT_MODEL_NAME, DEFAULT_MODEL_REVISION  # noqa: E402
from fast_dllm_relax.export_utils import (  # noqa: E402
    REQUIRED_FNS,
    load_hf_weights,
    make_relax_config,
    make_tvm_param,
    remap_params,
    write_kv_cache_manifest,
)
from fast_dllm_relax.model.metal import (  # noqa: E402
    METAL_STATIC_FNS,
    NARROW_I32_MAX_TOTAL_LEN,
    FastDLLMRelaxForCausalLMNative,
    metal_tir_pipeline,
)
from fast_dllm_relax.targets.rocm import (  # noqa: E402
    ROCM_HOST_TARGET_CONFIG,
    has_rdna_wmma,
    rocm_arch,
    rocm_smoke_test,
    rocm_target,
    rocm_target_config,
)

PIPELINE_NAME = "opt_fast_dllm_llm_native"
TORCH_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}
NP_DTYPES = {"float16": np.float16, "bfloat16": np.uint16}  # bf16 bits stored as uint16 (see dtype.txt)


def smoke_run(artifact_dir: Path, so_path: Path, relax_config, named_param_dict, dtype: str, dev) -> None:
    """Load on rocm:0 and check that every cache function writes only its own rows (as export_cuda.py)."""
    from fast_dllm_relax.model.cuda import build_prefix_block_attention_bias  # noqa: PLC0415
    from fast_dllm_relax.runtime import CacheSpec, allocate_prefix_cache  # noqa: PLC0415

    lib = tvm.runtime.load_module(str(so_path))
    vm = relax.VirtualMachine(lib, dev)
    param_arrays = np.load(artifact_dir / "fast_dllm_relax_nn_params.npz")
    params = [make_tvm_param(param_arrays[f"arr_{i}"], dtype, dev) for i in range(len(named_param_dict))]
    for fn in REQUIRED_FNS:
        vm[fn]
    print("  All required globals present")

    NL, HS = relax_config.num_hidden_layers, relax_config.hidden_size
    SBS, PLEN, BSIZ, CAP = 8, 32, 32, 96
    reshape = tvm.get_global_func("vm.builtin.reshape")

    def embed(n):
        ids = tvm.runtime.tensor(np.ones(n, dtype="int32"), dev)
        return reshape(vm["embed"](ids, params), tvm.runtime.ShapeTuple([1, n, HS]))

    spec = CacheSpec(NL, relax_config.num_key_value_heads, relax_config.head_dim, dtype)
    cache = allocate_prefix_cache(spec, dev, CAP)
    k_t = torch.from_dlpack(cache.k)  # zero-copy view of the TVM ROCm buffer (DLPack kDLROCM)
    k_t.fill_(7.0)
    bias = tvm.runtime.tensor(build_prefix_block_attention_bias(0, PLEN, block_size=BSIZ), dev)
    vm["build_prefix_cache"](embed(PLEN), bias, cache.k, cache.v, params)
    dev.sync()
    assert not (k_t[:, :, :PLEN] == 7.0).all(), "build_prefix_cache: rows [0, P) not written"
    assert (k_t[:, :, PLEN:] == 7.0).all(), "build_prefix_cache: wrote past prefix_len"
    assert torch.isfinite(k_t[:, :, :PLEN].float()).all(), "build_prefix_cache: non-finite K"
    print(f"  build_prefix_cache: wrote rows [0, {PLEN}) of a cap={CAP} cache")

    plen_arg = tvm.runtime.ShapeTuple([PLEN])
    off_blk = tvm.runtime.tensor(np.asarray(PLEN, dtype="int32"), dev)
    prefix_before = k_t[:, :, :PLEN].clone()
    h = vm["forward_cached"](embed(BSIZ), cache.k, cache.v, off_blk, plen_arg, params)
    dev.sync()
    assert tuple(h.shape) == (1, BSIZ, HS), "forward_cached: bad hidden shape"
    assert torch.equal(k_t[:, :, :PLEN], prefix_before), "forward_cached: modified the prefix"
    assert not (k_t[:, :, PLEN:PLEN + BSIZ] == 7.0).all(), "forward_cached: block rows not written"
    assert (k_t[:, :, PLEN + BSIZ:] == 7.0).all(), "forward_cached: wrote past the block"
    print(f"  forward_cached: h={tuple(h.shape)}, wrote rows [{PLEN}, {PLEN + BSIZ})")

    for rp in (0, 8, 16, 24):
        fn_name = f"forward_block_partial_rp{rp}"
        before = k_t.clone()
        off_rp = tvm.runtime.tensor(np.asarray(PLEN + rp, dtype="int32"), dev)
        k_t[:, :, PLEN + rp:PLEN + rp + SBS] = 7.0
        h2 = vm[fn_name](embed(SBS), cache.k, cache.v, off_rp, plen_arg, params)
        dev.sync()
        assert tuple(h2.shape) == (1, SBS, HS), f"{fn_name}: bad hidden shape"
        outside = torch.ones(k_t.shape[2], dtype=torch.bool, device=k_t.device)
        outside[PLEN + rp:PLEN + rp + SBS] = False
        assert torch.equal(k_t[:, :, outside], before[:, :, outside]), f"{fn_name}: wrote outside its window"
        assert not (k_t[:, :, PLEN + rp:PLEN + rp + SBS] == 7.0).all(), f"{fn_name}: window not written"
        print(f"  {fn_name}: h={tuple(h2.shape)}, wrote rows [{PLEN + rp}, {PLEN + rp + SBS})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION,
                        help="HF snapshot revision of --model (default: the pinned, validated one)")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--arch", default=None, help="AMDGPU arch (default: FAST_DLLM_ROCM_ARCH or gfx1151)")
    parser.add_argument("--warp-size", type=int, default=None, choices=[32, 64],
                        help="TVM thread_warp_size (default: 32 on RDNA, 64 on CDNA)")
    parser.add_argument("--gemm-acc", default="float32", choices=["float16", "float32"],
                        help="accumulator dtype of the projection/lm_head GEMMs (bfloat16 always float32)")
    parser.add_argument("--index-narrow-i32", action="store_true",
                        help="narrow device-kernel indexing to int32 (caps total length, as on Metal)")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--no-smoke", action="store_true", help="skip the on-device smoke run")
    parser.add_argument("--wmma", action=argparse.BooleanOptionalAction, default=None,
                        help="projections on the RDNA WMMA kernels (default: on for gfx11)")
    args = parser.parse_args()
    if args.dtype == "bfloat16":
        args.gemm_acc = "float32"

    arch = args.arch or rocm_arch()
    if args.wmma is None:
        args.wmma = has_rdna_wmma(arch)
    dtype = args.dtype
    tag = "fp16" if dtype == "float16" else "bf16"
    artifact_dir = REPO_ROOT / "artifacts" / f"fast_dllm_relax_{tag}_rocm_{arch}{args.suffix}"

    dev = tvm.rocm(0)
    print(f"Checking ROCm toolchain ({arch}) ...")
    rocm_smoke_test(arch)
    print("  ROCm smoke kernel OK\n")

    print(f"Loading {args.model}@{args.revision} (export dtype={dtype}) ...")
    t0 = time.perf_counter()
    hf_config, state_dict = load_hf_weights(args.model, args.revision)
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s")

    relax_config = make_relax_config(hf_config)
    relax_model = FastDLLMRelaxForCausalLMNative(relax_config, dtype=dtype, fused_qkv=True, fused_gate_up=True)
    if args.wmma:
        from fast_dllm_relax.model.rocm_linear import PRESETS, swap_projections  # noqa: PLC0415

        swap_projections(relax_model, relax_config, dtype, PRESETS["fp16"])
    mod, named_params = relax_model.export_tvm(spec=relax_model.get_default_spec())
    named_param_dict = dict(named_params)
    exported_fns = sorted(gv.name_hint for gv in mod.get_global_vars() if isinstance(mod[gv], relax.Function))
    print(f"  Exported functions: {exported_fns}")
    missing_fns = sorted(set(REQUIRED_FNS + METAL_STATIC_FNS) - set(exported_fns))
    if missing_fns:
        raise RuntimeError(f"native module is missing functions: {missing_fns}")
    if "fast_dllm." in mod.script() or "call_inplace_packed" in mod.script():
        raise RuntimeError("native module calls a fast_dllm.* Python callback")

    mapped = remap_params(state_dict, named_param_dict, relax_config,
                          torch_dtype=TORCH_DTYPES[dtype], np_dtype=NP_DTYPES[dtype])
    print(f"  Mapped {len(mapped)} params  (dtype={dtype})")

    target = rocm_target(arch, args.warp_size)
    print(f"\nCompiling with {PIPELINE_NAME}, target={target} host={target.host} ...")
    t1 = time.perf_counter()
    ex = tvm.compile(mod, target, relax_pipeline=relax.get_pipeline(PIPELINE_NAME, gemm_acc=args.gemm_acc),
                     tir_pipeline=metal_tir_pipeline(args.index_narrow_i32))
    compile_s = time.perf_counter() - t1
    print(f"  Compiled in {compile_s:.1f}s")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    so_path = artifact_dir / "fast_dllm_relax_nn.so"
    ex.export_library(str(so_path))
    param_list = [mapped[n] for n in named_param_dict]
    np.savez(artifact_dir / "fast_dllm_relax_nn_params.npz", *param_list)
    (artifact_dir / "fast_dllm_relax_nn_param_names.txt").write_text("\n".join(named_param_dict) + "\n", encoding="utf-8")
    (artifact_dir / "fast_dllm_relax_nn_dtype.txt").write_text(dtype + "\n", encoding="utf-8")
    write_kv_cache_manifest(artifact_dir, relax_config, dtype)
    print(f"  Saved: {so_path}")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tvm_version": tvm.__version__,
        "llvm_version": tvm.support.libinfo().get("LLVM_VERSION"),
        "target_device": rocm_target_config(arch, args.warp_size),
        "target_host": ROCM_HOST_TARGET_CONFIG,
        "pipeline": PIPELINE_NAME,
        "wmma": bool(args.wmma),
        "gemm_acc": args.gemm_acc,
        "device_index_dtype": "int32" if args.index_narrow_i32 else "int64",
        "max_total_len": NARROW_I32_MAX_TOTAL_LEN if args.index_narrow_i32 else None,
        "build_host": {"system": platform.system(), "machine": platform.machine()},
        "model_class": "fast_dllm_relax.model.metal.FastDLLMRelaxForCausalLMNative",
        "model_kwargs": {"fused_qkv": True, "fused_gate_up": True},
        "exported_functions": exported_fns,
        "num_params": len(param_list),
        "param_bytes": int(sum(a.nbytes for a in param_list)),
        "dtype": dtype,
        "hf_model": args.model,
        "hf_revision": args.revision,
        "compile_seconds": round(compile_s, 1),
    }
    (artifact_dir / "fast_dllm_relax_nn_rocm_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not args.no_smoke:
        print("\nVerifying artifact on rocm:0 ...")
        smoke_run(artifact_dir, so_path, relax_config, named_param_dict, dtype, dev)

    print(f"\n=== COMPLETE: {dtype} artifact for ROCm {arch}: {artifact_dir} ===")
    print(f"  python -m fast_dllm_relax.cli --backend relax --use-block-cache --device rocm:0 "
          f"--artifact-dir {artifact_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
