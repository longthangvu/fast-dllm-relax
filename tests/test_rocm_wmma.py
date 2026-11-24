"""RDNA (gfx11) WMMA GEMM kernels vs numpy: fp16, bf16, int8 (+ prefetch variant), int4.

Runs on a ROCm device with gfx11 WMMA (e.g. Radeon 8060S / gfx1151); skipped elsewhere. Covers decode
row counts (M = 8, 32) and a ragged dynamic M (300) with the tile settings the model uses.
"""

import numpy as np
import pytest
import tvm

from fast_dllm_relax.model import rocm_wmma as W

_has_rocm = tvm.rocm(0).exist


def _gfx11() -> bool:
    if not _has_rocm:
        return False
    return str(tvm.rocm(0).compute_version).startswith("11.")


pytestmark = pytest.mark.skipif(not _gfx11(), reason="needs a ROCm gfx11 (RDNA 3 / 3.5) GPU")

N, K, G = 256, 512, 128
ROWS = [8, 32, 300]


def _run(prim, args, m, out_dtype):
    from fast_dllm_relax.targets.rocm import rocm_target  # noqa: PLC0415

    f = tvm.compile(tvm.IRModule({"main": prim.with_attr("global_symbol", "main")}), target=rocm_target())["main"]
    dev = tvm.rocm(0)
    out = tvm.runtime.empty((m, N), out_dtype, dev)
    f(*[tvm.runtime.tensor(a, dev) for a in args], out)
    dev.sync()
    return out.numpy().astype("float32")


def _x(m, dtype="float16"):
    x = np.random.default_rng(m).uniform(-1, 1, (m, K)).astype("float32")
    if dtype == "bfloat16":
        import ml_dtypes  # noqa: PLC0415

        return x.astype(ml_dtypes.bfloat16)
    return x.astype(dtype)


@pytest.mark.parametrize("m", ROWS)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("tiles", [(2, 2, 64), (1, 1, 128)])
def test_wmma_gemm_float(m, dtype, tiles):
    x = _x(m, dtype)
    w = np.random.default_rng(1).uniform(-1, 1, (N, K)).astype(x.dtype)
    kernel = W.wmma_gemm_bf16 if dtype == "bfloat16" else W.wmma_gemm_f16
    y = _run(kernel(N, K, tiles[0], tiles[1], BK=tiles[2]), [x, w], m, dtype)
    ref = x.astype("float32") @ w.astype("float32").T
    tol = 3e-2 if dtype == "bfloat16" else 5e-3
    np.testing.assert_allclose(y, ref, rtol=tol, atol=tol * np.abs(ref).max())


@pytest.mark.parametrize("m", ROWS)
@pytest.mark.parametrize("kernel,tiles", [(W.wmma_gemm_w8, (2, 4, 64)), (W.wmma_gemm_w8, (1, 1, 128)),
                                          (W.wmma_gemm_w8_pf, (1, 1, 128)), (W.wmma_gemm_w8_pf, (1, 2, 128))])
def test_wmma_gemm_w8(m, kernel, tiles):
    rng = np.random.default_rng(2)
    q = rng.integers(0, 256, (N, K), dtype=np.uint8)
    scale = rng.uniform(0.001, 0.01, (N, K // G)).astype("float16")
    x = _x(m)
    y = _run(kernel(N, K, tiles[0], tiles[1], BK=tiles[2]), [x, q.view("<u4"), scale], m, "float16")
    w = (q.astype("float32") - 128) * np.repeat(scale.astype("float32"), G, axis=1)
    ref = x.astype("float32") @ w.T
    np.testing.assert_allclose(y, ref, rtol=1e-2, atol=1e-2 * np.abs(ref).max())


@pytest.mark.parametrize("m", ROWS)
@pytest.mark.parametrize("tiles", [(2, 2), (1, 1)])
def test_wmma_gemm_w4(m, tiles):
    rng = np.random.default_rng(3)
    q = rng.integers(0, 16, (N, K), dtype=np.uint32)
    packed = (q.reshape(N, K // 8, 8) << (np.arange(8, dtype=np.uint32) * 4)).sum(-1).astype(np.uint32)
    scale = rng.uniform(0.001, 0.02, (N, K // G)).astype("float16")
    x = _x(m)
    y = _run(W.wmma_gemm_w4(N, K, *tiles), [x, packed, scale], m, "float16")
    w = (q.astype("float32") - 8) * np.repeat(scale.astype("float32"), G, axis=1)
    ref = x.astype("float32") @ w.T
    np.testing.assert_allclose(y, ref, rtol=1e-2, atol=1e-2 * np.abs(ref).max())


def test_tile_guard_rejects_partial_loads():
    with pytest.raises(ValueError, match="whole 128-thread loads"):
        W.wmma_gemm_w8(N, K, 1, 1, BK=32)
