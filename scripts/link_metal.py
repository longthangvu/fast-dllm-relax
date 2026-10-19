#!/usr/bin/env python3
"""Run ON THE MAC: link a cross-compiled Metal artifact and check it loads.

scripts/export_metal.py ships `fast_dllm_relax_nn.tar` (Mach-O arm64 host objects + devc.o
carrying MSL source). This untars it, links `fast_dllm_relax_nn.so` with the local toolchain
(Xcode Command Line Tools clang), then loads it on tvm.device("metal", 0) and resolves all nine
exported VM functions. Shader compile errors only surface at first kernel launch -- run a
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
    "embed", "forward_hidden", "get_logits", "build_prefix_cache", "forward_cached",
    "forward_block_partial_rp0", "forward_block_partial_rp8",
    "forward_block_partial_rp16", "forward_block_partial_rp24",
)


def main() -> int:
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
