"""Load AMD Quark int4 (weight-only, group 128, symmetric) checkpoints into the RocmW4Linear layout.

Quark exports each quantized Linear in the AWQ "GEMM" packing:
  L.weight             int32 [K, N/8]  8 columns per word, nibble order (0, 4, 1, 5, 2, 6, 3, 7)
  L.weight_scale       f32   [K/G, N]
  L.weight_zero_point  int32 [K/G, N/8] (all zero for symmetric int4)
Symmetric int4 nibbles are signed two's complement (-8..7). RocmW4Linear expects unsigned nibbles with
zero point 8, so each nibble's top bit is flipped (q ^ 8).

Output per Linear: packed uint32 [N, K/8] (nibble i of word j = q[n, 8j + i]) and fp16 scales [N, K/G].
Pure torch / numpy; no vendor library needed.
"""

from pathlib import Path

import numpy as np
import torch

AWQ_NIBBLE_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)
GROUP = 128


def _unpack_int32_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """int32 [..., C] -> uint8 [..., 8C]; nibble i of word c goes to 8c + i."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    vals = (packed.to(torch.int32).unsqueeze(-1) >> shifts) & 0xF
    return vals.reshape(*packed.shape[:-1], packed.shape[-1] * 8).to(torch.uint8)


def _awq_unpack(packed: torch.Tensor) -> torch.Tensor:
    """AWQ-packed int32 [R, N/8] -> uint8 [R, N] in natural column order."""
    r, n = packed.shape[0], packed.shape[1] * 8
    order = torch.tensor(AWQ_NIBBLE_ORDER)
    return _unpack_int32_nibbles(packed).view(r, n // 8, 8)[:, :, order].reshape(r, n)


def pack_rows_uint4(q: torch.Tensor) -> np.ndarray:
    """uint4 values [N, K] -> uint32 [N, K/8] with nibble i of word j = q[n, 8j + i]."""
    n, k = q.shape
    q = q.to(torch.int64).view(n, k // 8, 8)
    shifts = torch.arange(0, 32, 4, dtype=torch.int64)
    words = (q << shifts).sum(-1)
    return np.ascontiguousarray(words.numpy().astype(np.uint32))


def quark_int4_to_tvm_w4(qweight: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor):
    """Convert one Quark int4 Linear to (packed uint32 [N, K/8], scales fp16 [N, K/G])."""
    if int(zero_point.abs().max()) != 0:
        raise ValueError("only symmetric int4 (zero point 0) is supported by RocmW4Linear")
    k, n = qweight.shape[0], qweight.shape[1] * 8
    if scale.shape[1] != n or k // scale.shape[0] != GROUP:
        raise ValueError(f"expected group {GROUP} scales [K/{GROUP}, N]; got {tuple(scale.shape)} for K={k}, N={n}")
    q = _awq_unpack(qweight).t() ^ 8  # [N, K], signed -> unsigned with zero point 8
    return pack_rows_uint4(q), scale.t().contiguous().to(torch.float16).numpy()


def dequantize_tvm_w4(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Reference dequantization of the RocmW4Linear layout -> fp32 [N, K] (for checks)."""
    n = packed.shape[0]
    shifts = np.arange(0, 32, 4, dtype=np.uint32)
    q = ((packed[..., None] >> shifts) & 0xF).reshape(n, -1).astype(np.float32)
    return (q - 8.0) * np.repeat(scales.astype(np.float32), GROUP, axis=1)


def load_quark_int4(quark_dir):
    """Read a Quark export directory.

    Returns (int4, other): int4 maps each int4-quantized Linear name to (packed, scales) in the
    RocmW4Linear layout; other maps every remaining tensor name (embedding, norms, biases, unquantized
    Linears) to a torch tensor as exported. Taking the non-quantized tensors from the export (not the
    original model) keeps algorithms that rewrite them (AWQ folds scales into the RMSNorm weights)
    correct.
    """
    from safetensors import safe_open  # noqa: PLC0415

    files = sorted(Path(quark_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {quark_dir}")
    int4, other = {}, {}
    for path in files:
        with safe_open(str(path), "pt") as f:
            keys = set(f.keys())
            quantized = {k[: -len(".weight_scale")] for k in keys if k.endswith(".weight_scale")}
            quantized = {n for n in quantized if f"{n}.weight_zero_point" in keys}
            for name in sorted(quantized):
                int4[name] = quark_int4_to_tvm_w4(
                    f.get_tensor(f"{name}.weight"),
                    f.get_tensor(f"{name}.weight_scale"),
                    f.get_tensor(f"{name}.weight_zero_point"),
                )
            q_keys = {f"{n}.{s}" for n in quantized for s in ("weight", "weight_scale", "weight_zero_point")}
            for key in sorted(keys - q_keys):
                other[key] = f.get_tensor(key)
    return int4, other
