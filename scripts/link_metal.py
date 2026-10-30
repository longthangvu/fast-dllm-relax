#!/usr/bin/env python3
"""Run ON THE MAC: link an exported Metal artifact, check it loads, and smoke-test its KV cache.

scripts/export_metal.py (on the workstation or the Mac) writes `fast_dllm_relax_nn.tar` (Mach-O
arm64 host objects + devc.o carrying MSL source). This untars it, links `fast_dllm_relax_nn.so`
with the local toolchain (Xcode Command Line Tools clang), then loads it on tvm.device("metal", 0),
resolves the required VM functions and reports the optional static-shape ones. Finally it loads the
artifact through the runner (weights + KV-cache manifest) and checks, on the device, that every
cache function writes exactly its own rows of the in-place KV cache. Shader compile errors surface
at first kernel launch, so this also compiles the cache path's kernels.

Usage (Mac, repo root, TVM env from `scripts/setup_tvm.sh mac`):
    python scripts/link_metal.py --artifact-dir artifacts/fast_dllm_relax_fp16_metal
"""

import argparse
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import tvm
from tvm import relax
from tvm.contrib import cc

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fast_dllm_relax.export_utils import KV_CACHE_MANIFEST, REQUIRED_FNS  # noqa: E402
from fast_dllm_relax.model.metal import METAL_STATIC_FNS as STATIC_FNS  # noqa: E402

SENTINEL = 7.0


def kv_cache_smoke_test(artifact_dir: Path) -> None:
    """Each cache function writes only its row window (mirrors scripts/export_cuda.py's smoke test,
    with TVM tensors read back through .numpy() instead of torch views)."""
    from fast_dllm_relax.model.cuda import build_prefix_block_attention_bias  # noqa: PLC0415
    from fast_dllm_relax.runners import FastDLLMRelaxRunner  # noqa: PLC0415
    from fast_dllm_relax.runtime import allocate_prefix_cache, static_fn_name  # noqa: PLC0415

    runner = FastDLLMRelaxRunner(artifact_dir=artifact_dir, device="metal:0", load_model=True)
    vm, params, dev, reshape = runner.vm, runner.params, runner.dev, runner.reshape
    hidden = int(runner.hf_config.hidden_size)
    bsiz, sbs, plen, cap = runner.block_size, runner.small_block_size, 32, 96

    def embed(n):
        ids = tvm.runtime.tensor(np.ones(n, dtype="int32"), dev)
        return reshape(vm["embed"](ids, params), tvm.runtime.ShapeTuple([1, n, hidden]))

    cache = allocate_prefix_cache(runner.cache_spec, dev, cap)
    host_dtype = cache.k.numpy().dtype
    for t in (cache.k, cache.v):
        t.copyfrom(np.full(t.shape, SENTINEL, dtype=host_dtype))
    untouched = lambda a: (a == SENTINEL).all()  # noqa: E731
    written = lambda a: not (a == SENTINEL).any()  # noqa: E731

    bias = tvm.runtime.tensor(build_prefix_block_attention_bias(0, plen, block_size=bsiz), dev)
    vm["build_prefix_cache"](embed(plen), bias, cache.k, cache.v, params)
    for t in (cache.k, cache.v):
        a = t.numpy()
        assert written(a[:, :, :plen]), "build_prefix_cache: rows [0, P) not written"
        assert untouched(a[:, :, plen:]), "build_prefix_cache: wrote past prefix_len"
    prefix_k = cache.k.numpy()[:, :, :plen].copy()

    off = tvm.runtime.tensor(np.asarray(plen, dtype="int32"), dev)
    plen_arg = tvm.runtime.ShapeTuple([plen])
    for fn in sorted({"forward_cached", static_fn_name(vm, "forward_cached", bsiz)}):
        h = vm[fn](embed(bsiz), cache.k, cache.v, off, plen_arg, params)
        a = cache.k.numpy()
        assert tuple(h.shape) == (1, bsiz, hidden), f"{fn}: bad hidden shape {tuple(h.shape)}"
        assert np.array_equal(a[:, :, :plen], prefix_k), f"{fn}: modified the prefix"
        assert written(a[:, :, plen:plen + bsiz]), f"{fn}: block rows not written"
        assert untouched(a[:, :, plen + bsiz:]), f"{fn}: wrote past the block"
        print(f"  {fn}: wrote rows [{plen}, {plen + bsiz})")

    for rp in range(0, bsiz, sbs):
        window = slice(plen + rp, plen + rp + sbs)
        host = cache.k.numpy()
        host[:, :, window] = SENTINEL
        cache.k.copyfrom(host)
        off_rp = tvm.runtime.tensor(np.asarray(plen + rp, dtype="int32"), dev)
        vm[f"forward_block_partial_rp{rp}"](embed(sbs), cache.k, cache.v, off_rp, plen_arg, params)
        a = cache.k.numpy()
        outside = np.ones(cap, dtype=bool)
        outside[window] = False
        assert np.array_equal(a[:, :, outside], host[:, :, outside]), f"rp{rp}: wrote outside its window"
        assert written(a[:, :, window]), f"rp{rp}: window not written"
    print(f"  forward_block_partial_rp{{0..{bsiz - sbs}}}: each wrote only its {sbs}-row window")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    args = parser.parse_args()
    artifact_dir = Path(args.artifact_dir)
    if not (artifact_dir / KV_CACHE_MANIFEST).exists():
        print(f"{artifact_dir} has no {KV_CACHE_MANIFEST}: re-export it with scripts/export_metal.py")
        return 1
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
    print("KV-cache smoke test (on device):")
    kv_cache_smoke_test(artifact_dir)
    print("KV-cache smoke test OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
