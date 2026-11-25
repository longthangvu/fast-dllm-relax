"""Projection layers for the ROCm build: fp16 / bf16, int8 and int4 weights on RDNA WMMA kernels.

The native model's fused projections (qkv, o, gate_up, down) can be swapped per projection type
(`swap_projections`) for:

- `RocmWmmaLinear`: fp16 / bf16 weights (same parameter names as nn.Linear).
- `RocmW8Linear`: int8 weight-only, group 128, round-to-nearest (zero point 128), fp32 accumulation.
- `RocmW4Linear`: int4 weight-only, group 128 (zero point 8), e.g. AMD Quark GPTQ checkpoints
  (fast_dllm_relax/quark_int4.py).

and the tied logits projection can use an int8 copy of the embedding (`add_w8_lm_head`).

All matmuls run the hand-written RDNA (gfx11) WMMA kernels in `rocm_wmma.py`. Dynamic-M calls
(prefill, scoring) use multi-row tiles; static-M calls (decode, M = 8 / 32) use the per-shape decode
tiles. The quantized weights are decoded into LDS once per block. `FAST_DLLM_ROCM_WMMA=0` at export
time keeps the portable paths: dlight GEMMs for fp16, and a fused TE dequant+matmul for int4. The int8
layers always need WMMA.

Weight layouts:
- int4: uint32 [N, K/8], nibble i of word j = q[n, 8j + i]
- int8: uint32 [N, K/4], byte i of word j = q[n, 4j + i]
- scales: [N, K/128] in the model dtype
"""

import os
import types

from tvm import te, tirx
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

GROUP = 128
PROJECTIONS = ("qkv", "o", "gate_up", "down")
PRECISIONS = ("f16", "w8", "w4")  # "f16" = the model dtype (fp16 or bf16)
PRESETS = {
    "fp16": dict(qkv="f16", o="f16", gate_up="f16", down="f16"),
    "int8-mlp": dict(qkv="f16", o="f16", gate_up="w8", down="w8"),
    "int4-mlp": dict(qkv="f16", o="f16", gate_up="w4", down="w4"),
}


def wmma_enabled() -> bool:
    return os.environ.get("FAST_DLLM_ROCM_WMMA", "1") == "1"


def _rows(x: Tensor):
    """(leading dims, flattened row count, whether that count is static)."""
    lead = list(x.shape[:-1])
    rows = 1
    for d in lead:
        rows = rows * d
    return lead, rows, isinstance(rows, (int, tirx.IntImm))


def te_w4_linear(x, packed, scale, bias=None):
    """Portable int4 path: x [M, K], packed uint32 [N, K/8], scale [N, K/GROUP] (, bias [N]) -> [M, N].

    One TE function, so dequantization stays fused with the reduction (as separate Relax ops the full
    fp16 weight would be materialized on every call). dlight schedules it with ROCM_INT4_MATMUL_CONFIGS.
    """
    m, k = x.shape
    n = packed.shape[0]

    def deq(i, j):
        nib = (packed[i, j // 8] >> ((j % 8) * 4).astype("uint32")) & tirx.const(15, "uint32")
        return (nib.astype("float16") - tirx.const(8, "float16")) * scale[i, j // GROUP]

    w = te.compute((n, k), deq, name="dequantize")
    r = te.reduce_axis((0, k), name="k")
    acc = te.compute((m, n), lambda i, j: te.sum(x[i, r].astype("float32") * w[j, r].astype("float32"), axis=r),
                     name="matmul")
    if bias is None:
        return te.compute((m, n), lambda i, j: acc[i, j].astype("float16"), name="cast")
    return te.compute((m, n), lambda i, j: acc[i, j].astype("float16") + bias[j], name="cast_bias")


class RocmWmmaLinear(nn.Module):
    """fp16 / bf16 Linear on the RDNA WMMA kernel; same parameter names as nn.Linear."""

    def __init__(self, in_features: int, out_features: int, bias: bool, dtype: str):
        self.in_features, self.out_features, self.dtype = in_features, out_features, dtype
        self.weight = nn.Parameter((out_features, in_features), dtype)
        self.bias = nn.Parameter((out_features,), dtype) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        from .rocm_wmma import WMMA_F16_DECODE_TILES, WMMA_F16_TILES, wmma_gemm_bf16, wmma_gemm_f16, wmma_tiles  # noqa: PLC0415

        lead, rows, static_rows = _rows(x)
        decode_tiles = None
        if static_rows:
            decode_tiles = WMMA_F16_DECODE_TILES.get((int(rows), self.out_features, self.in_features))
            # bf16 has no fast SIMT path on gfx11 (per-element bf16 -> fp32), so it always uses WMMA.
            if decode_tiles is None and self.dtype == "bfloat16":
                decode_tiles = (1, 4, 64)
        if not wmma_enabled() or (static_rows and decode_tiles is None):
            y = op.matmul(x, op.permute_dims(self.weight))  # dlight path
            return y + self.bias if self.bias is not None else y
        if static_rows:
            ti, tj, bk = decode_tiles
        else:
            (ti, tj), bk = wmma_tiles(self.out_features, self.in_features, WMMA_F16_TILES), 64
        kernel = wmma_gemm_bf16 if self.dtype == "bfloat16" else wmma_gemm_f16
        y = op.tensor_ir_op(
            kernel(self.out_features, self.in_features, ti, tj, BK=bk), f"{self.dtype}_wmma",
            args=[op.reshape(x, (rows, self.in_features)), self.weight],
            out=Tensor.placeholder((rows, self.out_features), x.dtype),
        )
        if self.bias is not None:
            y = y + self.bias
        return op.reshape(y, (*lead, self.out_features))


class RocmW8Linear(nn.Module):
    """y = x @ dequant(W)^T (+ bias), int8 g128 weights, RDNA WMMA kernel for all M."""

    def __init__(self, in_features: int, out_features: int, bias: bool, dtype: str):
        self.in_features, self.out_features = in_features, out_features
        self.weight_packed = nn.Parameter((out_features, in_features // 4), "uint32")
        self.weight_scales = nn.Parameter((out_features, in_features // GROUP), dtype)
        self.bias = nn.Parameter((out_features,), dtype) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        from .rocm_wmma import WMMA_W8_DECODE_TILES, wmma_gemm_w8, wmma_gemm_w8_pf  # noqa: PLC0415

        lead, rows, static_rows = _rows(x)
        kernel, (ti, tj, bk) = wmma_gemm_w8, (2, 4, 64)
        if static_rows:
            ti, tj, bk = WMMA_W8_DECODE_TILES.get((int(rows), self.out_features, self.in_features), (1, 4, 64))
            if self.in_features > self.out_features:  # long-K decode GEMM (down_proj): prefetch hides latency
                kernel = wmma_gemm_w8_pf
        y = op.tensor_ir_op(kernel(self.out_features, self.in_features, ti, tj, BK=bk), "w8_linear",
                            args=[op.reshape(x, (rows, self.in_features)), self.weight_packed, self.weight_scales],
                            out=Tensor.placeholder((rows, self.out_features), x.dtype))
        if self.bias is not None:
            y = y + self.bias
        return op.reshape(y, (*lead, self.out_features))


class RocmW4Linear(nn.Module):
    """y = x @ dequant(W)^T (+ bias), int4 g128 weights.

    wmma=True (RDNA): the WMMA kernel for dynamic M and for the decode shapes in WMMA_W4_DECODE_TILES.
    Otherwise (and on other targets, e.g. Metal) the portable fused TE function."""

    def __init__(self, in_features: int, out_features: int, bias: bool, dtype: str, wmma: bool = False):
        self.in_features, self.out_features = in_features, out_features
        self.wmma = wmma
        self.weight_packed = nn.Parameter((out_features, in_features // 8), "uint32")
        self.weight_scales = nn.Parameter((out_features, in_features // GROUP), dtype)
        self.bias = nn.Parameter((out_features,), dtype) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        lead, rows, static_rows = _rows(x)
        x2 = op.reshape(x, (rows, self.in_features))
        tiles = None
        if self.wmma and wmma_enabled():
            from .rocm_wmma import WMMA_W4_DECODE_TILES, wmma_tiles  # noqa: PLC0415

            if static_rows:
                tiles = WMMA_W4_DECODE_TILES.get((int(rows), self.out_features, self.in_features))
            else:
                tiles = wmma_tiles(self.out_features, self.in_features)
        if tiles is not None:
            from .rocm_wmma import wmma_gemm_w4  # noqa: PLC0415

            y = op.tensor_ir_op(
                wmma_gemm_w4(self.out_features, self.in_features, *tiles), "w4_wmma",
                args=[x2, self.weight_packed, self.weight_scales],
                out=Tensor.placeholder((rows, self.out_features), x.dtype),
            )
            if self.bias is not None:
                y = y + self.bias
        else:
            args = [x2, self.weight_packed, self.weight_scales] + ([self.bias] if self.bias is not None else [])
            y = op.tensor_expr_op(te_w4_linear, "w4_linear", args)
        return op.reshape(y, (*lead, self.out_features))


def projection_shapes(cfg) -> dict:
    """(in_features, out_features, bias) of each fused projection type."""
    q_out = (cfg.num_attention_heads + 2 * cfg.num_key_value_heads) * cfg.head_dim
    return {
        "qkv": (cfg.hidden_size, q_out, True),
        "o": (cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, False),
        "gate_up": (cfg.hidden_size, 2 * cfg.intermediate_size, False),
        "down": (cfg.intermediate_size, cfg.hidden_size, False),
    }


def swap_projections(model, cfg, dtype: str, precision: dict, wmma: bool = True) -> None:
    """Replace each decoder layer's fused projections by precision: {qkv, o, gate_up, down} ->
    "f16" (model dtype) | "w8" | "w4". wmma=False keeps fp16 projections as nn.Linear and int4 on
    the portable TE path (non-RDNA targets); int8 needs wmma."""
    if set(precision) != set(PROJECTIONS) or not set(precision.values()) <= set(PRECISIONS):
        raise ValueError(f"precision must map {PROJECTIONS} to {PRECISIONS}, got {precision}")
    if not wmma and "w8" in precision.values():
        raise ValueError("int8 projections need the RDNA WMMA kernels (wmma=True)")
    shapes = projection_shapes(cfg)
    for layer in model.model.layers:
        for proj in PROJECTIONS:
            owner = layer.self_attn if proj in ("qkv", "o") else layer.mlp
            i, o, b = shapes[proj]
            kind = precision[proj]
            if kind == "f16":
                if not wmma:
                    continue  # keep the model's own nn.Linear
                module = RocmWmmaLinear(i, o, b, dtype)
            elif kind == "w8":
                module = RocmW8Linear(i, o, b, dtype)
            else:
                module = RocmW4Linear(i, o, b, dtype, wmma)
            setattr(owner, f"{proj}_proj", module)


def quantize_rows_int8_g128(w):
    """Symmetric int8 (zero point 128), group 128 along K, round-to-nearest.
    w [N, K] torch tensor -> (uint32 [N, K/4], fp16 scales [N, K/128]) numpy arrays."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    n, k = w.shape
    g = w.float().view(n, k // GROUP, GROUP)
    s = (g.abs().amax(-1) / 127).clamp_min(1e-8)
    q = (torch.round(g / s[..., None]).clamp(-127, 127) + 128).to(torch.uint8).view(n, k)
    return np.ascontiguousarray(q.cpu().numpy()).view("<u4"), s.to(torch.float16).cpu().numpy()


class RocmW8LMHead(nn.Module):
    """int8 g128 copy of the (tied) output projection for the logits, RDNA WMMA kernel."""

    def __init__(self, hidden: int, vocab: int, dtype: str):
        self.hidden, self.vocab = hidden, vocab
        self.weight_packed = nn.Parameter((vocab, hidden // 4), "uint32")
        self.weight_scales = nn.Parameter((vocab, hidden // GROUP), dtype)

    def forward(self, x: Tensor) -> Tensor:
        from .rocm_wmma import wmma_gemm_w8  # noqa: PLC0415

        lead, rows, static_rows = _rows(x)
        ti, tj = (1, 4) if static_rows else (2, 4)
        y = op.tensor_ir_op(wmma_gemm_w8(self.vocab, self.hidden, ti, tj), "w8_lm_head",
                            args=[op.reshape(x, (rows, self.hidden)), self.weight_packed, self.weight_scales],
                            out=Tensor.placeholder((rows, self.vocab), x.dtype))
        return op.reshape(y, (*lead, self.vocab))


def add_w8_lm_head(model, cfg, dtype: str) -> None:
    """Route the logits (all logits / confidence paths go through get_logits) through an int8 copy
    of the tied embedding; parameters lm_head_w8.weight_packed / weight_scales."""
    model.lm_head_w8 = RocmW8LMHead(cfg.hidden_size, cfg.vocab_size, dtype)

    def get_logits(self, hidden_states: Tensor) -> Tensor:
        logits = self.lm_head_w8(hidden_states)
        return logits.astype("float32") if logits.dtype != "float32" else logits

    model.get_logits = types.MethodType(get_logits, model)
