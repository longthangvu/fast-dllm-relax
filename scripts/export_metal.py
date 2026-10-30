#!/usr/bin/env python3
"""Cross-compile the Fast-dLLM v2 Relax artifact for a Mac (Apple silicon, Metal), float16.

Builds `FastDLLMRelaxForCausalLMNative` (`fast_dllm_relax/model/metal.py`: in-graph attention, no
fast_dllm.flash_attn_bf16 callback, plus static-shape variants) with the `opt_fast_dllm_llm_native`
pipeline (dlight only, no cuBLAS/CUTLASS, no CUDA graph) for Metal (MSL source) + an
arm64-apple-macos host. Device kernels index in int32 by default (`--no-index-narrow-i32` keeps
int64); such an artifact caps prompt + generation at NARROW_I32_MAX_TOTAL_LEN tokens, recorded in
the manifest and enforced by the runner. Writes:

  artifacts/fast_dllm_relax_fp16_metal[<suffix>]/
      fast_dllm_relax_nn.tar              Mach-O arm64 host objects + devc.o (MSL source)
      fast_dllm_relax_nn_params.npz / _param_names.txt / _dtype.txt
      fast_dllm_relax_nn_metal_manifest.json

Runs either on a Linux workstation whose TVM was built with configs/tvm/cuda.cmake (USE_METAL=OFF
plus the patched build_metal_off.cc, LLVM with the AArch64 target), or on the Mac itself
(configs/tvm/mac.cmake). Checks before writing: Mach-O arm64 members, MSL source present,
per-kernel buffer-arg / threadgroup-memory limits, no 64-bit indexing left in a narrowed build.
Link the directory on the Mac with scripts/link_metal.py (copy it there first if built elsewhere).

Usage (repo root, TVM env from scripts/setup_tvm.sh):
    python scripts/export_metal.py
"""

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
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
    remap_params,
)
from fast_dllm_relax.model.metal import (  # noqa: E402
    METAL_STATIC_FNS,
    NARROW_I32_MAX_TOTAL_LEN,
    FastDLLMRelaxForCausalLMNative,
    metal_tir_pipeline,
)
from fast_dllm_relax.targets.metal import (  # noqa: E402
    METAL_HOST_TARGET_CONFIG,
    METAL_TARGET_CONFIG,
    check_metal_cross_toolchain,
    inspect_metal_tar,
    metal_smoke_test,
    metal_target,
)

PIPELINE_NAME = "opt_fast_dllm_llm_native"
TORCH_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}
NP_DTYPES = {"float16": np.float16, "bfloat16": np.uint16}  # bf16 bits stored as uint16 (see dtype.txt)
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "fast_dllm_relax_fp16_metal"


def register_metallib_compiler() -> bool:
    """Register tvm_callback_metal_compile so BuildMetal emits metallib instead of MSL source.

    Matches the runtime's source compile: fast math on; MSL 3.1 for kernels that use bfloat, 2.3
    otherwise. Returns False (nothing registered) when the Metal compiler is unavailable."""
    if shutil.which("xcrun") is None:
        return False
    if subprocess.run(["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True).returncode != 0:
        return False

    @tvm.register_global_func("tvm_callback_metal_compile", override=True)
    def _compile(code: str, _target) -> bytes:
        std = "-std=metal3.1" if "bfloat" in code else "-std=macos-metal2.3"
        with tempfile.TemporaryDirectory() as tmp:
            src, air, lib = (Path(tmp) / n for n in ("k.metal", "k.air", "k.metallib"))
            src.write_text(code, encoding="utf-8")
            subprocess.run(["xcrun", "-sdk", "macosx", "metal", std, "-ffast-math", "-O3", "-c", str(src), "-o", str(air)],
                           check=True)
            subprocess.run(["xcrun", "-sdk", "macosx", "metallib", str(air), "-o", str(lib)], check=True)
            return lib.read_bytes()

    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION,
                        help="HF snapshot revision of --model (default: the pinned, validated one)")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"],
                        help="weight/activation dtype. bfloat16 kernels compile as MSL 3.1 at load "
                             "(macOS 14+; native bf16 arithmetic on M2 and newer); float16 as MSL 2.3.")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--no-index-narrow-i32", dest="index_narrow_i32", action="store_false",
                        help="keep int64 device-kernel indexing (slower; no total-length cap)")
    parser.add_argument("--gemm-acc", default=None, choices=["float16", "float32"],
                        help="accumulator dtype of the projection/lm_head GEMMs (default float32; "
                             "float16 reproduces the pre-plan-002 fp16-accumulation artifact; "
                             "bfloat16 always accumulates in float32)")
    parser.add_argument("--metallib", action="store_true",
                        help="precompile kernels to metallib with `xcrun metal` (Mac with Xcode only; "
                             "without it the artifact keeps MSL source, compiled at load)")
    args = parser.parse_args()
    if args.dtype == "bfloat16":
        if args.gemm_acc == "float16":
            parser.error("--dtype bfloat16 accumulates in float32")
        args.gemm_acc = "float32"
    elif args.gemm_acc is None:
        args.gemm_acc = "float32"

    dtype = args.dtype
    narrow_i32 = args.index_narrow_i32
    base_dir = ARTIFACT_DIR if dtype == "float16" else ARTIFACT_DIR.with_name(ARTIFACT_DIR.name.replace("fp16", "bf16"))
    artifact_dir = base_dir.with_name(base_dir.name + args.suffix) if args.suffix else base_dir

    print("Checking Metal cross toolchain ...")
    check_metal_cross_toolchain()
    metal_smoke_test(artifact_dir)
    print("  cross toolchain OK\n")

    print(f"Loading {args.model}@{args.revision} (export dtype={dtype}) ...")
    t0 = time.perf_counter()
    hf_config, state_dict = load_hf_weights(args.model, args.revision)
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s")

    relax_config = make_relax_config(hf_config)
    relax_model = FastDLLMRelaxForCausalLMNative(relax_config, dtype=dtype, fused_qkv=True, fused_gate_up=True)
    mod, named_params = relax_model.export_tvm(spec=relax_model.get_default_spec())
    named_param_dict = dict(named_params)
    exported_fns = sorted(gv.name_hint for gv in mod.get_global_vars()
                          if isinstance(mod[gv], relax.Function))
    print(f"  Exported functions: {exported_fns}")
    missing_fns = sorted(set(REQUIRED_FNS + METAL_STATIC_FNS) - set(exported_fns))
    if missing_fns:
        raise RuntimeError(f"native module is missing functions: {missing_fns}")
    if "flash_attn" in mod.script():
        raise RuntimeError("native module references flash_attn")

    mapped = remap_params(state_dict, named_param_dict, relax_config,
                          torch_dtype=TORCH_DTYPES[dtype], np_dtype=NP_DTYPES[dtype])
    print(f"  Mapped {len(mapped)} params  (dtype={dtype})")

    metal_output = "MSL source (tvm_callback_metal_compile not registered)"
    if args.metallib:
        if register_metallib_compiler():
            metal_output = "metallib (xcrun metal, -ffast-math; MSL 3.1 for bfloat kernels, else 2.3)"
        else:
            print("  --metallib: `xcrun metal` not found (needs Xcode); keeping MSL source")

    target = metal_target()
    print(f"\nCompiling with {PIPELINE_NAME}, target={target} host={target.host}, "
          f"device index dtype={'int32' if narrow_i32 else 'int64'} ...")
    t1 = time.perf_counter()
    ex = tvm.compile(mod, target, relax_pipeline=relax.get_pipeline(PIPELINE_NAME, gemm_acc=args.gemm_acc),
                     tir_pipeline=metal_tir_pipeline(narrow_i32))
    compile_s = time.perf_counter() - t1
    print(f"  Compiled in {compile_s:.1f}s")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    lib_path = artifact_dir / "fast_dllm_relax_nn.tar"
    # No fcompile: the .tar holds host .o files + devc.o (MSL source); it is linked on the Mac.
    ex.export_library(str(lib_path))
    param_list = [mapped[n] for n in named_param_dict]
    np.savez(artifact_dir / "fast_dllm_relax_nn_params.npz", *param_list)
    (artifact_dir / "fast_dllm_relax_nn_param_names.txt").write_text(
        "\n".join(named_param_dict) + "\n", encoding="utf-8"
    )
    (artifact_dir / "fast_dllm_relax_nn_dtype.txt").write_text(dtype + "\n", encoding="utf-8")
    print(f"  Saved: {lib_path}")

    print("\nInspecting Metal tar ...")
    tar_info = inspect_metal_tar(lib_path)
    print(f"  members={tar_info['members']}")
    print(f"  kernels={tar_info['num_kernels']} max_buffer_index={tar_info['max_buffer_index']} "
          f"max_threadgroup_bytes={tar_info['max_threadgroup_bytes']} long_count={tar_info['long_count']}")
    if narrow_i32 and tar_info["long_count"]:
        raise RuntimeError(f"int32-narrowed build still has {tar_info['long_count']} 64-bit integer uses in its MSL")

    param_dtypes = sorted({str(a.dtype) for a in param_list})
    if param_dtypes != [np.dtype(NP_DTYPES[dtype]).name]:
        raise RuntimeError(f"Unexpected param dtypes in {dtype} Metal artifact: {param_dtypes}")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tvm_version": tvm.__version__,
        "llvm_version": tvm.support.libinfo().get("LLVM_VERSION"),
        "target_device": METAL_TARGET_CONFIG,
        "target_host": METAL_HOST_TARGET_CONFIG,
        "metal_output": metal_output,
        "pipeline": PIPELINE_NAME,
        "gemm_acc": args.gemm_acc,
        "tir_pipeline": "s_tir+NarrowDeviceIndexToInt32" if narrow_i32 else "default",
        "device_index_dtype": "int32" if narrow_i32 else "int64",
        # Read by the runner: requests whose prompt + generation + block padding exceed it are rejected.
        "max_total_len": NARROW_I32_MAX_TOTAL_LEN if narrow_i32 else None,
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
        "metal_tar_inspection": tar_info,
    }
    manifest_path = artifact_dir / "fast_dllm_relax_nn_metal_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"  Wrote manifest: {manifest_path}")
    print(f"\n=== COMPLETE: {dtype} artifact compiled for Metal: {artifact_dir} ===")
    if platform.system() != "Darwin":
        print("Copy the directory to the Mac (same relative path), then from the repo root there:")
    else:
        print("Link it (repo root):")
    print(f"  python scripts/link_metal.py --artifact-dir {artifact_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
