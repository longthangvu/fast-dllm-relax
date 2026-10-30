#!/usr/bin/env python3
"""Run ON THE MAC: link an exported Metal artifact and check it loads.

scripts/export_metal.py (on the workstation or the Mac) writes `fast_dllm_relax_nn.tar` (Mach-O
arm64 host objects + devc.o carrying MSL source). This untars it, links `fast_dllm_relax_nn.so`
with the local toolchain (Xcode Command Line Tools clang), then loads it on tvm.device("metal", 0),
resolves all ten required VM functions and reports the optional static-shape ones. Shader compile errors only surface at first kernel launch -- run a
`python -m fast_dllm_relax.cli --backend relax --device metal:0 ...` smoke run afterwards.

Usage (Mac, repo root, TVM env from `scripts/setup_tvm.sh mac`):
    python scripts/link_metal.py --artifact-dir artifacts/fast_dllm_relax_fp16_metal
"""

import argparse
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import tvm
from tvm import relax
from tvm.contrib import cc

REQUIRED_FNS = (
    "embed", "forward_hidden", "get_logits", "build_prefix_cache", "forward_cached", "forward_cached_block",
    "forward_block_partial_rp0", "forward_block_partial_rp8",
    "forward_block_partial_rp16", "forward_block_partial_rp24",
)
# fast_dllm_relax.model.metal.METAL_STATIC_FNS: optional (the runtime falls back to the dynamic
# functions without them), reported so an artifact exported before they existed is easy to spot.
STATIC_FNS = (
    "forward_cached_b32", "forward_cached_block_b32", "get_logits_m8", "get_logits_m32",
    "get_confidence_m8", "get_confidence_m32",
)


_NOT_PORTED = (
    "The Metal model is not ported to the in-place KV cache yet (plan 003: "
    ".codex/thoughts/shared/plans/003_inplace_kv_cache_and_q1_fix.md), and the runner no longer "
    "loads functional-cache artifacts."
)


def main() -> int:
    raise SystemExit(_NOT_PORTED)
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    args = parser.parse_args()
    artifact_dir = Path(args.artifact_dir)
    tar_path = artifact_dir / "fast_dllm_relax_nn.tar"
    so_path = artifact_dir / "fast_dllm_relax_nn.so"

    dev = tvm.device("metal", 0)
    print(f"tvm.device('metal', 0).exist = {dev.exist}")
    if not dev.exist:
        print("Metal device not visible from this shell (e.g. a non-GUI SSH session); aborting.")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(tar_path) as tf:
            names = tf.getnames()
            tf.extractall(tmp, filter="data")
        objs = [str(Path(tmp) / n) for n in names]
        print(f"linking {len(objs)} objects -> {so_path}")
        cc.create_shared(str(so_path), objs)
    print(subprocess.run(["file", str(so_path)], check=True, text=True, capture_output=True).stdout.strip())

    lib = tvm.runtime.load_module(str(so_path))
    vm = relax.VirtualMachine(lib, dev)
    missing = []
    for fn in REQUIRED_FNS:
        try:
            vm[fn]
        except Exception:  # noqa: BLE001
            missing.append(fn)
    if missing:
        print(f"MISSING globals: {missing}")
        return 1
    print(f"All {len(REQUIRED_FNS)} required globals present. Link + load OK.")
    present = []
    for fn in STATIC_FNS:
        try:
            vm[fn]
            present.append(fn)
        except Exception:  # noqa: BLE001
            pass
    absent = [fn for fn in STATIC_FNS if fn not in present]
    print(f"Static-shape globals: {len(present)}/{len(STATIC_FNS)} present"
          + (f" (absent: {absent}; the runtime uses the dynamic functions instead)" if absent else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
