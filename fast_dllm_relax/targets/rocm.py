"""AMD ROCm target helpers (native build and run on the same machine, e.g. Strix Halo gfx1151).

TVM generates AMDGPU code through LLVM (the TVM build must link an LLVM that knows the arch, see
configs/tvm/rocm.cmake). RDNA GPUs (gfx10xx/11xx/12xx) execute wave32 by default, while TVM's
`rocm` target kind defaults to thread_warp_size=64 (CDNA / Instinct). The warp size decides how
dlight lays out warp-level reductions, so RDNA targets set it to 32 explicitly.
"""

import os

import tvm
from tvm.script import tirx as T

# Strix Halo (Ryzen AI Max, Radeon 8060S). Override with FAST_DLLM_ROCM_ARCH, e.g. gfx1150 for
# Ryzen AI 300/400 iGPUs (Radeon 880M/890M).
DEFAULT_ROCM_ARCH = "gfx1151"
ROCM_HOST_TARGET_CONFIG = {"kind": "llvm", "mtriple": "x86_64-linux-gnu"}


def rocm_arch() -> str:
    return os.environ.get("FAST_DLLM_ROCM_ARCH", DEFAULT_ROCM_ARCH)


def has_rdna_wmma(arch: str) -> bool:
    """gfx11 (RDNA 3 / 3.5) WMMA, as used by model/rocm_wmma.py. gfx12 WMMA uses a different operand
    layout and is not supported by those kernels."""
    return arch.startswith("gfx11")


def default_warp_size(arch: str) -> int:
    """32 on RDNA (gfx10xx, gfx11xx, gfx12xx), 64 on GCN/CDNA (gfx9xx)."""
    return 32 if arch.startswith(("gfx10", "gfx11", "gfx12")) else 64


def rocm_target_config(arch: str | None = None, warp_size: int | None = None) -> dict:
    arch = arch or rocm_arch()
    return {
        "kind": "rocm",
        "mcpu": arch,
        "mtriple": "amdgcn-amd-amdhsa-hcc",
        "thread_warp_size": warp_size or default_warp_size(arch),
        "max_num_threads": 1024,
        "max_threads_per_block": 1024,
        "max_shared_memory_per_block": 65536,
    }


def rocm_target(arch: str | None = None, warp_size: int | None = None) -> tvm.target.Target:
    return tvm.target.Target(rocm_target_config(arch, warp_size), host=ROCM_HOST_TARGET_CONFIG)


def rocm_smoke_test(arch: str | None = None) -> None:
    """Compile and run a vector add on rocm:0; raises if the toolchain or device is broken."""
    import numpy as np  # noqa: PLC0415

    dev = tvm.rocm(0)
    if not dev.exist:
        raise RuntimeError("no ROCm device visible to TVM (check /dev/kfd permissions and the TVM build)")
    lib = tvm.compile(tvm.IRModule({"main": smoke_vec_add}), target=rocm_target(arch))
    n = 1024
    a = tvm.runtime.tensor(np.arange(n, dtype="float32"), dev)
    b = tvm.runtime.tensor(np.ones(n, dtype="float32"), dev)
    c = tvm.runtime.tensor(np.zeros(n, dtype="float32"), dev)
    lib["main"](a, b, c)
    if not np.allclose(c.numpy(), np.arange(n, dtype="float32") + 1):
        raise RuntimeError("ROCm smoke kernel produced wrong results")


@T.prim_func
def smoke_vec_add(a: T.Buffer((1024,), "float32"), b: T.Buffer((1024,), "float32"),
                  c: T.Buffer((1024,), "float32")) -> None:
    T.func_attr({"global_symbol": "main", "tirx.noalias": True})
    for bx in T.thread_binding(4, thread="blockIdx.x"):
        for tx in T.thread_binding(256, thread="threadIdx.x"):
            c[bx * 256 + tx] = a[bx * 256 + tx] + b[bx * 256 + tx]
