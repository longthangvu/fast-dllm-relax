"""Jetson Orin cross-compile helpers.

Target/host construction, cross-toolchain preflight check, and a tiny AArch64 smoke test,
used by every export script that supports `--target jetson_orin`
(`scripts/export_cuda.py`).
"""

import shutil
import subprocess
from pathlib import Path

import tvm
from tvm.contrib import cc
from tvm.script import from_source

JETSON_HOST_TARGET_CONFIG = {"kind": "llvm", "mtriple": "aarch64-linux-gnu", "mcpu": "cortex-a78"}
JETSON_CUDA_TARGET_CONFIG = {
    "kind": "cuda",
    "arch": "sm_87",
    "max_shared_memory_per_block": 49152,
    "max_threads_per_block": 1024,
    "thread_warp_size": 32,
    "registers_per_block": 65536,
}
JETSON_CROSS_CXX = "aarch64-linux-gnu-g++"
JETSON_CROSS_CC = "aarch64-linux-gnu-gcc"


def jetson_target() -> tvm.target.Target:
    return tvm.target.Target(JETSON_CUDA_TARGET_CONFIG, host=JETSON_HOST_TARGET_CONFIG)


def check_jetson_cross_toolchain() -> None:
    """Fail fast if the AArch64 cross toolchain isn't usable, before any model work runs."""
    missing = [tool for tool in (JETSON_CROSS_CXX, JETSON_CROSS_CC, "file") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            f"Missing required tool(s) for --target jetson_orin: {missing}. "
            "Install with: sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu"
        )
    llvm_config = shutil.which("llvm-config")
    if llvm_config is None:
        raise RuntimeError("llvm-config is required to confirm this LLVM build includes AArch64 support.")
    llvm_targets = subprocess.run(
        [llvm_config, "--targets-built"], check=True, text=True, capture_output=True
    ).stdout
    if "AArch64" not in llvm_targets.split():
        raise RuntimeError(
            f"LLVM was not built with AArch64 support. llvm-config --targets-built returned: {llvm_targets}"
        )


def jetson_smoke_test(artifact_dir: Path) -> str:
    """Compile+cross-link a tiny AArch64 CPU module before any real model work.

    Returns the `file` command output for the smoke artifact. Raises if the cross-linked
    library isn't actually AArch64 — catches a broken cross toolchain before the ~10 min
    real export runs.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    smoke_so = artifact_dir / "smoke_aarch64.so"
    smoke_source = r'''
# from tvm.script import tirx as T
@T.prim_func
def smoke_vec_add(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16,), "float32")
    B = T.match_buffer(b, (16,), "float32")
    C = T.match_buffer(c, (16,), "float32")
    T.func_attr({"global_symbol": "smoke_vec_add", "tirx.noalias": True})
    for i in T.grid(16):
        C[i] = A[i] + B[i]
'''
    smoke_prim_func = from_source(smoke_source)
    smoke_rt_mod = tvm.compile(smoke_prim_func, target=tvm.target.Target(JETSON_HOST_TARGET_CONFIG))
    smoke_rt_mod.export_library(str(smoke_so), fcompile=cc.cross_compiler(JETSON_CROSS_CXX))

    file_output = subprocess.run(["file", str(smoke_so)], check=True, text=True, capture_output=True).stdout.strip()
    print(f"  smoke test: {file_output}")
    if "ARM aarch64" not in file_output:
        raise RuntimeError(f"Smoke artifact is not AArch64: {file_output}")
    return file_output
