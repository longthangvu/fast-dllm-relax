#!/usr/bin/env python3
"""Export the Fast-dLLM v2 Relax artifact for CUDA (desktop GPU or Jetson Orin cross-compile).

Builds `FastDLLMRelaxForCausalLM` (fused QKV + gate/up, per-layer block-cache partial forwards)
with the `opt_fast_dllm_llm_cublas` pipeline and writes:

  artifacts/fast_dllm_relax_{bf16,fp16}[_jetson_orin][<suffix>]/
      fast_dllm_relax_nn.so               compiled library (VM executable)
      fast_dllm_relax_nn_params.npz       weights (bf16 stored as uint16 bits)
      fast_dllm_relax_nn_param_names.txt
      fast_dllm_relax_nn_dtype.txt

--target jetson_orin cross-compiles for Jetson Orin (sm_87, aarch64-linux-gnu host). It needs
aarch64-linux-gnu-g++/-gcc and an LLVM built with the AArch64 target on this machine. The
cross-compiled .so cannot be loaded here, so the post-export smoke run is replaced by file/readelf
AArch64 checks; copy the directory to the Jetson and run there.

Usage (repo root, TVM env from scripts/setup_tvm.sh, nvcc on PATH):
    python scripts/export_cuda.py                          # bf16, this machine's GPU
    python scripts/export_cuda.py --dtype float16
    python scripts/export_cuda.py --dtype float16 --target jetson_orin
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import tvm
from tvm import relax
from tvm.contrib import cc

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fast_dllm_relax.common import DEFAULT_MODEL_NAME, DEFAULT_MODEL_REVISION  # noqa: E402
from fast_dllm_relax.export_utils import (  # noqa: E402
    REQUIRED_FNS,
    check_byoc_offload,
    load_hf_weights,
    make_relax_config,
    make_tvm_param,
    remap_params,
)
from fast_dllm_relax.model.cuda import FastDLLMRelaxForCausalLM  # noqa: E402
import fast_dllm_relax.runtime  # noqa: E402,F401  registers the fast_dllm.flash_attn_bf16 packed fn

_ARTIFACT_DIR_BY_DTYPE = {
    "bfloat16": REPO_ROOT / "artifacts" / "fast_dllm_relax_bf16",
    "float16":  REPO_ROOT / "artifacts" / "fast_dllm_relax_fp16",
}
_TORCH_DTYPE = {
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
}
_NP_DTYPE = {
    "float16":  np.float16,
    "bfloat16": np.uint16,  # bf16 bits stored as uint16; dtype.txt records the semantic dtype
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION,
                        help="HF snapshot revision of --model (default: the pinned, validated one)")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"],
                        help="Weight and compute dtype for the exported artifact (default: bfloat16, "
                             "the checkpoint's native dtype)")
    parser.add_argument("--suffix", default="",
                        help="Appended to the artifact directory name so a one-off export doesn't "
                             "overwrite the default artifact.")
    parser.add_argument("--target", default="native", choices=["native", "jetson_orin"],
                        help="'native' compiles for this machine's CUDA device (default). "
                             "'jetson_orin' cross-compiles for Jetson Orin (sm_87, aarch64-linux-gnu "
                             "host); the result cannot be loaded on this machine.")
    args = parser.parse_args()

    dtype = args.dtype
    torch_dtype = _TORCH_DTYPE[dtype]
    np_dtype = _NP_DTYPE[dtype]
    artifact_dir = _ARTIFACT_DIR_BY_DTYPE[dtype]
    if args.target == "jetson_orin":
        artifact_dir = artifact_dir.with_name(artifact_dir.name + "_jetson_orin")
    if args.suffix:
        artifact_dir = artifact_dir.with_name(artifact_dir.name + args.suffix)

    if args.target == "jetson_orin":
        from fast_dllm_relax.targets import jetson  # noqa: PLC0415

        print("Checking AArch64 cross toolchain ...")
        jetson.check_jetson_cross_toolchain()
        jetson.jetson_smoke_test(artifact_dir)
        print("  cross toolchain OK\n")

    print(f"Loading {args.model}@{args.revision} (export dtype={dtype}) ...")
    t0 = time.perf_counter()
    hf_config, state_dict = load_hf_weights(args.model, args.revision)
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s")

    relax_config = make_relax_config(hf_config)
    print(f"Config: {relax_config.num_hidden_layers}L × {relax_config.hidden_size}H, "
          f"{relax_config.num_attention_heads}Q / {relax_config.num_key_value_heads}KV heads, "
          f"head_dim={relax_config.head_dim}")

    print(f"\nExporting Relax model (dtype={dtype}) ...")
    relax_model = FastDLLMRelaxForCausalLM(
        relax_config, dtype=dtype, fused_qkv=True, fused_gate_up=True
    )
    mod, named_params = relax_model.export_tvm(spec=relax_model.get_default_spec())
    named_param_dict = dict(named_params)
    print(f"  Exported functions: {[str(gv) for gv in mod.functions.keys()]}")

    print("\nRemapping HF weights ...")
    mapped = remap_params(state_dict, named_param_dict, relax_config,
                          torch_dtype=torch_dtype, np_dtype=np_dtype)
    print(f"  Mapped {len(mapped)} params  (dtype={dtype})")

    print("\nCompiling with opt_fast_dllm_llm_cublas pipeline ...")
    if args.target == "jetson_orin":
        dev = None
        target = jetson.jetson_target()
        print(f"  target: {target}  (cross-compiling for Jetson Orin, host={target.host})")
    else:
        dev = tvm.device("cuda", 0)
        target = tvm.target.Target.from_device(dev)
    # CUTLASS's codegen compiles via a plain, non-cross-compile-aware nvcc invocation, so it is
    # skipped when cross-compiling; the batched attention matmuls it would claim fall back to dlight.
    use_cutlass = args.target != "jetson_orin"
    pipeline = relax.get_pipeline("opt_fast_dllm_llm_cublas", use_cutlass=use_cutlass)
    t1 = time.perf_counter()
    ex = tvm.compile(mod, target, relax_pipeline=pipeline)
    print(f"  Compiled in {time.perf_counter()-t1:.1f}s")
    module_kinds = check_byoc_offload(ex)

    print(f"\nSaving to {artifact_dir} ...")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    so_path = artifact_dir / "fast_dllm_relax_nn.so"
    # libflash_attn.so (built with TVM when present) is linked defensively for native bf16
    # exports; the compiled model does not reference its symbols at this shape. Skipped for Jetson:
    # it is x86_64-only, and an x86_64 .so cannot be linked into an aarch64 cross-compile.
    _flash_attn_so = REPO_ROOT / "3rdparty/tvm/3rdparty/libflash_attn/src/libflash_attn.so"
    # --no-as-needed forces the library into NEEDED even when it appears before the .o files
    # that reference it (the default --as-needed would drop it due to link-order processing).
    _extra_link = [
        "-Wl,--no-as-needed",
        str(_flash_attn_so),
        "-Wl,--as-needed",
        f"-Wl,-rpath,{_flash_attn_so.parent}",
    ] if (dtype == "bfloat16" and _flash_attn_so.exists() and args.target != "jetson_orin") else []
    if args.target == "jetson_orin":
        ex.export_library(str(so_path), fcompile=cc.cross_compiler(jetson.JETSON_CROSS_CXX), options=_extra_link)
    else:
        ex.export_library(str(so_path), options=_extra_link)
    np.savez(artifact_dir / "fast_dllm_relax_nn_params.npz",
             *[mapped[n] for n in named_param_dict])
    (artifact_dir / "fast_dllm_relax_nn_param_names.txt").write_text(
        "\n".join(named_param_dict) + "\n", encoding="utf-8"
    )
    (artifact_dir / "fast_dllm_relax_nn_dtype.txt").write_text(dtype + "\n", encoding="utf-8")
    print(f"  Saved: {so_path}")

    if args.target == "jetson_orin":
        print("\nVerifying artifact (AArch64 checks only — a cross-compiled .so cannot load here) ...")
        file_output = subprocess.run(["file", str(so_path)], check=True, text=True, capture_output=True).stdout.strip()
        print(f"  {file_output}")
        if "ARM aarch64" not in file_output:
            raise RuntimeError(f"Artifact is not AArch64: {file_output}")
        readelf_machine = None
        readelf_path = shutil.which("readelf")
        if readelf_path:
            readelf_output = subprocess.run(
                [readelf_path, "-h", str(so_path)], check=True, text=True, capture_output=True
            ).stdout
            for line in readelf_output.splitlines():
                if "Machine:" in line:
                    readelf_machine = line.split(":", 1)[1].strip()
                    break
            print(f"  readelf machine: {readelf_machine}")
            if readelf_machine != "AArch64":
                raise RuntimeError(f"readelf did not report AArch64: {readelf_machine}")
        else:
            print("  readelf unavailable; skipping ELF header machine assertion")

        cross_compiler_version = subprocess.run(
            [jetson.JETSON_CROSS_CXX, "--version"], check=True, text=True, capture_output=True
        ).stdout.splitlines()[0]
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tvm_version": tvm.__version__,
            "target": str(target),
            "target_host": str(target.host),
            "cuda_arch": jetson.JETSON_CUDA_TARGET_CONFIG["arch"],
            "host_mtriple": jetson.JETSON_HOST_TARGET_CONFIG["mtriple"],
            "host_mcpu": jetson.JETSON_HOST_TARGET_CONFIG["mcpu"],
            "cross_compiler": shutil.which(jetson.JETSON_CROSS_CXX),
            "cross_compiler_version": cross_compiler_version,
            "dtype": dtype,
            "module_kinds": module_kinds,
            "file_output": file_output,
            "readelf_machine": readelf_machine,
            "param_count": len(named_param_dict),
            "source_model": args.model,
            "source_revision": args.revision,
        }
        (artifact_dir / "fast_dllm_relax_nn_jetson_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"  Wrote manifest: {artifact_dir / 'fast_dllm_relax_nn_jetson_manifest.json'}")

        print(f"\n=== COMPLETE: {dtype} artifact cross-compiled for Jetson Orin ===")
        print(f"  Artifact: {artifact_dir}")
        print("Copy this directory to the Jetson (same relative path), then check there:")
        print("  ldd fast_dllm_relax_nn.so    # expect: no \"not found\" entries")
        return

    print("\nVerifying artifact ...")
    if _flash_attn_so.exists():
        import ctypes  # noqa: PLC0415

        ctypes.CDLL(str(_flash_attn_so))  # pre-load so libflash_attn symbols resolve
    lib = tvm.runtime.load_module(str(so_path))
    vm = relax.VirtualMachine(lib, dev)
    param_arrays = np.load(artifact_dir / "fast_dllm_relax_nn_params.npz")
    params = [
        make_tvm_param(param_arrays[f"arr_{i}"], dtype, dev)
        for i in range(len(named_param_dict))
    ]
    for fn in REQUIRED_FNS:
        vm[fn]
    print("  All required globals present")

    # Smoke test: per-layer partial forward (all 4 replace_position variants)
    NL = relax_config.num_hidden_layers
    NKV = relax_config.num_key_value_heads
    HD = relax_config.head_dim
    SBS = 8; PLEN = 32; BSIZ = 32
    HS = relax_config.hidden_size
    reshape = tvm.get_global_func("vm.builtin.reshape")

    tok_ids = tvm.runtime.tensor(np.ones(SBS, dtype='int32'), dev)
    emb = vm['embed'](tok_ids, params)
    emb_r = reshape(emb, tvm.runtime.ShapeTuple([1, SBS, HS]))
    pk = tvm.runtime.empty((NL, 1, NKV, PLEN, HD), dtype, dev)
    pv = tvm.runtime.empty((NL, 1, NKV, PLEN, HD), dtype, dev)
    bk = tvm.runtime.empty((NL, 1, NKV, BSIZ, HD), dtype, dev)
    bv = tvm.runtime.empty((NL, 1, NKV, BSIZ, HD), dtype, dev)
    for rp in (0, 8, 16, 24):
        fn_name = f"forward_block_partial_rp{rp}"
        off_rp = tvm.runtime.tensor(np.asarray(PLEN + rp, dtype='int32'), dev)
        h2, nak, nbk, nbv = vm[fn_name](emb_r, pk, pv, bk, bv, off_rp, params)
        assert tuple(h2.shape) == (1, SBS, HS), f"{fn_name}: bad hidden shape"
        assert tuple(nak.shape) == (NL, 1, NKV, SBS, HD), f"{fn_name}: bad nak shape"
        assert tuple(nbk.shape) == (NL, 1, NKV, BSIZ, HD), f"{fn_name}: bad nbk shape"
        assert tuple(nbv.shape) == (NL, 1, NKV, BSIZ, HD), f"{fn_name}: bad nbv shape"
        print(f"  {fn_name}: h={tuple(h2.shape)}, nbk={tuple(nbk.shape)}")

    print(f"\n=== COMPLETE: {dtype} artifact ready ===")
    print(f"  Artifact: {artifact_dir}")


if __name__ == "__main__":
    main()
