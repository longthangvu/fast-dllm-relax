"""Mac (Apple silicon, Metal) cross-compile helpers.

Mirrors `targets/jetson.py`: target construction, toolchain preflight, a tiny smoke build,
plus `inspect_metal_tar` for the workstation-side checks of an exported `.tar`.

The Metal device code is emitted as MSL *source* (no `tvm_callback_metal_compile` registered) and
compiled by the Mac's Metal runtime at module load. The host code is Mach-O arm64 objects packed
into a `.tar`; the final link to `fast_dllm_relax_nn.so` happens on the Mac
(`scripts/link_metal.py`) because a Linux workstation has no macOS SDK/ld64.
"""

import re
import shutil
import struct
import subprocess
import tarfile
import tempfile
from pathlib import Path

import tvm
from tvm.script import from_source

METAL_TARGET_CONFIG = {
    "kind": "metal",
    "max_threads_per_block": 1024,
    "max_shared_memory_per_block": 32768,
    "thread_warp_size": 32,
    "max_function_args": 31,
}
# apple-m4 needs LLVM >= 19; apple-m3 is the newest CPU LLVM 18 knows.
# Only affects the host-side VM glue code, not the GPU kernels.
METAL_HOST_TARGET_CONFIG = {"kind": "llvm", "mtriple": "arm64-apple-macos", "mcpu": "apple-m3"}

_MAX_BUFFER_INDEX = METAL_TARGET_CONFIG["max_function_args"] - 1
# Must match kMetalModuleVersion in src/runtime/metal/metal_module.mm (identical on the Mac).
_METAL_MODULE_VERSION = "0.1.0"
_MAX_THREADGROUP_BYTES = METAL_TARGET_CONFIG["max_shared_memory_per_block"]
_MSL_TYPE_BYTES = {
    "half": 2, "bfloat": 2, "float": 4, "int": 4, "uint": 4, "short": 2, "ushort": 2,
    "char": 1, "uchar": 1, "bool": 1, "long": 8, "ulong": 8,
    "half2": 4, "half4": 8, "float2": 8, "float4": 16, "int2": 8, "int4": 16, "uint4": 16,
}


def metal_target() -> tvm.target.Target:
    return tvm.target.Target(METAL_TARGET_CONFIG, host=METAL_HOST_TARGET_CONFIG)


def check_metal_cross_toolchain() -> None:
    """Fail fast if this TVM/LLVM can't produce Mac Metal artifacts, before any model work runs."""
    missing = [tool for tool in ("file", "strings") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"Missing required tool(s) for the Metal cross-compile: {missing}")
    llvm_config = shutil.which("llvm-config")
    if llvm_config is None:
        raise RuntimeError("llvm-config is required to confirm this LLVM build includes AArch64 support.")
    llvm_targets = subprocess.run(
        [llvm_config, "--targets-built"], check=True, text=True, capture_output=True
    ).stdout
    if "AArch64" not in llvm_targets.split():
        raise RuntimeError(f"LLVM was not built with AArch64 support: {llvm_targets}")
    if tvm.get_global_func("target.build.metal", allow_missing=True) is None:
        raise RuntimeError("This TVM build has no Metal codegen (target.build.metal missing).")
    if tvm.get_global_func("tvm_callback_metal_compile", allow_missing=True) is not None:
        raise RuntimeError(
            "tvm_callback_metal_compile is registered: Metal codegen would emit a metallib instead of "
            "MSL source, which a non-Mac host cannot produce correctly for the Mac."
        )


def inspect_metal_tar(tar_path: Path) -> dict:
    """Workstation-side checks of an exported Metal `.tar`. Raises on any hard failure."""
    tar_path = Path(tar_path)
    result: dict = {"members": {}}
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(tar_path) as tf:
            names = tf.getnames()
            tf.extractall(tmp, filter="data")
        for name in names:
            desc = subprocess.run(
                ["file", "-b", str(Path(tmp) / name)], check=True, text=True, capture_output=True
            ).stdout.strip()
            result["members"][name] = desc
            if "Mach-O 64-bit" not in desc or "arm64" not in desc:
                raise RuntimeError(f"{name} is not a Mach-O arm64 object: {desc}")
        if "devc.o" not in names:
            raise RuntimeError(f"No devc.o (device blob) in {tar_path}: {names}")

        devc_raw = (Path(tmp) / "devc.o").read_bytes()
        devc = devc_raw.decode("latin-1")
        if "metal_stdlib" not in devc:
            raise RuntimeError("devc.o has no '#include <metal_stdlib>' -- not MSL source")
        # The Mac's ffi.Module.load_from_bytes.metal reads (version, smap, fmap, fmt). A stock
        # USE_METAL=OFF build serializes a generic DeviceSourceModule (fmt, fmap, source) instead,
        # which fails on the Mac with std::bad_alloc -- needs the build_metal_off.cc patch (patches/).
        if (struct.pack("<Q", len(_METAL_MODULE_VERSION)) + _METAL_MODULE_VERSION.encode()) not in devc_raw:
            raise RuntimeError(
                "devc.o Metal blob is not in MetalModuleNode layout (no length-prefixed version "
                f"'{_METAL_MODULE_VERSION}'); rebuild libtvm with the patched src/target/opt/build_metal_off.cc"
            )
        host_text = "".join(
            (Path(tmp) / n).read_bytes().decode("latin-1") for n in names if n != "devc.o"
        )
        if "flash_attn" in host_text:
            raise RuntimeError("host objects reference flash_attn -- callback leaked into the Metal build")

    # Per-kernel resource checks on the MSL source.
    kernels = re.split(r"(?=kernel void )", devc)[1:]
    max_buffer = {}
    tg_bytes = {}
    for body in kernels:
        name_match = re.match(r"kernel void (\w+)", body)
        kname = name_match.group(1) if name_match else "<unknown>"
        # Signature ends at the first '{'.
        sig, _, rest = body.partition("{")
        bufs = [int(b) for b in re.findall(r"\[\[\s*buffer\((\d+)\)\s*\]\]", sig)]
        max_buffer[kname] = max(bufs) if bufs else -1
        # Function body ends where the next top-level kernel starts (split above); good enough for
        # summing static threadgroup arrays declared inside it.
        total = 0
        for ty, dims in re.findall(r"threadgroup\s+(\w+)\s+\w+((?:\[\d+\])+)", rest):
            count = 1
            for d in re.findall(r"\[(\d+)\]", dims):
                count *= int(d)
            total += count * _MSL_TYPE_BYTES.get(ty, 4)
        tg_bytes[kname] = total

    result["num_kernels"] = len(kernels)
    result["max_buffer_index"] = max(max_buffer.values()) if max_buffer else -1
    result["max_threadgroup_bytes"] = max(tg_bytes.values()) if tg_bytes else 0
    over_buf = {k: v for k, v in max_buffer.items() if v > _MAX_BUFFER_INDEX}
    over_tg = {k: v for k, v in tg_bytes.items() if v > _MAX_THREADGROUP_BYTES}
    result["kernels_over_buffer_limit"] = over_buf
    result["kernels_over_threadgroup_limit"] = over_tg
    if over_buf:
        raise RuntimeError(f"Kernels exceed Metal buffer-arg limit (> {_MAX_BUFFER_INDEX}): {over_buf}")
    if over_tg:
        raise RuntimeError(f"Kernels exceed threadgroup memory ({_MAX_THREADGROUP_BYTES} B): {over_tg}")
    return result


def metal_smoke_test(artifact_dir: Path) -> dict:
    """Compile a tiny GPU kernel for the Metal target and export it as a `.tar` before the real
    export runs; catches a broken toolchain before the slow model compile."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    smoke_tar = artifact_dir / "smoke_metal.tar"
    smoke_source = r'''
# from tvm.script import tirx as T
@T.prim_func
def smoke_vec_add(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (1024,), "float16")
    B = T.match_buffer(b, (1024,), "float16")
    C = T.match_buffer(c, (1024,), "float16")
    T.func_attr({"global_symbol": "smoke_vec_add", "tirx.noalias": True})
    for bx in T.thread_binding(8, thread="blockIdx.x"):
        for tx in T.thread_binding(128, thread="threadIdx.x"):
            C[bx * 128 + tx] = A[bx * 128 + tx] + B[bx * 128 + tx]
'''
    smoke_prim_func = from_source(smoke_source)
    rt_mod = tvm.compile(smoke_prim_func, target=metal_target())
    rt_mod.export_library(str(smoke_tar))
    info = inspect_metal_tar(smoke_tar)
    print(f"  smoke test: {info['members']}, kernels={info['num_kernels']}")
    smoke_tar.unlink()
    return info
