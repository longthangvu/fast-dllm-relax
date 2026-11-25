"""Quark int4 (AWQ-packed, signed nibbles, g128) -> RocmW4Linear layout conversion (CPU only)."""

import numpy as np
import pytest
import torch

from fast_dllm_relax.quark_int4 import AWQ_NIBBLE_ORDER, GROUP, dequantize_tvm_w4, quark_int4_to_tvm_w4


def _awq_pack(q: torch.Tensor) -> torch.Tensor:
    """uint4 [R, N] -> int32 [R, N/8] in AWQ nibble order (what Quark exports)."""
    r, n = q.shape
    q = q.to(torch.int64).view(r, n // 8, 8)
    words = torch.zeros(r, n // 8, dtype=torch.int64)
    for col, slot in enumerate(AWQ_NIBBLE_ORDER):  # natural column col lives in nibble slot
        words |= q[:, :, col] << (4 * slot)
    return torch.from_numpy(words.numpy().astype(np.uint32).view(np.int32))


def _make_quark(n: int, k: int, seed: int = 0):
    """Random symmetric int4 Linear in Quark's format, plus its integer values [N, K]."""
    g = torch.Generator().manual_seed(seed)
    signed = torch.randint(-8, 8, (k, n), generator=g)  # [K, N], two's-complement nibbles in the file
    scale = torch.rand(k // GROUP, n, generator=g) * 0.02 + 1e-3
    zero_point = torch.zeros(k // GROUP, n // 8, dtype=torch.int32)
    return _awq_pack(signed & 0xF), scale, zero_point, signed.t()


@pytest.mark.parametrize("n, k", [(64, 128), (256, 512)])
def test_quark_int4_roundtrip(n, k):
    qweight, scale, zero_point, ints = _make_quark(n, k)
    packed, scales = quark_int4_to_tvm_w4(qweight, scale, zero_point)
    assert packed.shape == (n, k // 8) and packed.dtype == np.uint32
    assert scales.shape == (n, k // GROUP) and scales.dtype == np.float16
    np.testing.assert_array_equal(scales, scale.t().to(torch.float16).numpy())
    expected = ints.numpy().astype(np.float32) * np.repeat(scales.astype(np.float32), GROUP, axis=1)
    np.testing.assert_array_equal(dequantize_tvm_w4(packed, scales), expected)


def test_quark_int4_rejects_asymmetric():
    qweight, scale, zero_point, _ = _make_quark(64, 128)
    zero_point[0, 0] = 1
    with pytest.raises(ValueError, match="symmetric"):
        quark_int4_to_tvm_w4(qweight, scale, zero_point)
