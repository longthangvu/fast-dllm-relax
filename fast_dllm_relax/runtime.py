"""Runtime helpers for Fast-dLLM Relax prompt scoring and generation."""

from __future__ import annotations

import dataclasses
import math as _math
import os
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as _F
import tvm

from .common import release_relax_memory_pool
from .model.cuda import build_prefix_block_attention_bias, extract_shifted_block_logits

# ── attention / KV-write packed functions ─────────────────────────────────────
# Registered at import time so they're available before any VM is created. The CUDA model calls them
# via rx.call_inplace_packed, once per layer, as synchronous VM-to-Python callbacks:
#   fast_dllm.attn_inplace  forward_cached / forward_block_partial_rp*: write the active rows' K/V
#                           into the in-place cache, then attend over prefix + block.
#   fast_dllm.kv_write      build_prefix_cache: write the prefix rows' K/V.
# The cache is [num_layers, 1, cap, num_kv_heads, head_dim] (BSNH rows); q/k/v are BSNH. Row
# windows arrive as host ints (prefix_len plus compile-time or shape-derived offsets), so nothing
# syncs the device. Writes and attention run on torch's current stream, as the attention always has.
_FLASH_SCALE = 1.0 / _math.sqrt(128)  # head_dim=128

try:
    import flash_attn as _flash_attn

    def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """BSNH attention, no mask; returns a contiguous BSNH tensor."""
        return _flash_attn.flash_attn_func(q, k, v, softmax_scale=_FLASH_SCALE, causal=False).contiguous()

except ImportError:
    # Fallback: PyTorch SDPA dispatches to its native FlashAttention-2 kernel when attn_mask=None.
    # enable_gqa=True attends on the native num_kv_heads tensors; bitwise-identical to an explicit
    # repeat-then-attend at this model's shapes/dtype.

    def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """BSNH attention, no mask; returns a contiguous BSNH tensor."""
        out = _F.scaled_dot_product_attention(  # SDPA expects BHSD
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            scale=_FLASH_SCALE, is_causal=False, enable_gqa=True,
        )
        return out.transpose(1, 2).contiguous()  # BHSD → BSNH


def _write_rows(k_nd, v_nd, k_cache_nd, v_cache_nd, layer: int, start: int) -> tuple[torch.Tensor, torch.Tensor]:
    k_cache = torch.from_dlpack(k_cache_nd)
    v_cache = torch.from_dlpack(v_cache_nd)
    k_new = torch.from_dlpack(k_nd)
    rows = k_new.shape[1]
    if start < 0 or start + rows > k_cache.shape[2]:
        raise ValueError(f"KV write [{start}, {start + rows}) outside cache capacity {k_cache.shape[2]}")
    k_cache[layer, :, start : start + rows].copy_(k_new)
    v_cache[layer, :, start : start + rows].copy_(torch.from_dlpack(v_nd))
    return k_cache, v_cache


@tvm.register_global_func("fast_dllm.attn_inplace", override=True)
def _attn_inplace_packed(q_nd, k_nd, v_nd, k_cache_nd, v_cache_nd, layer, prefix_len, write_rel, attend_rel):
    """Write k/v at cache rows [prefix_len + write_rel, + rows), attend q over rows [0, prefix_len + attend_rel)."""
    write_start = prefix_len + write_rel
    attend_len = prefix_len + attend_rel
    rows = k_nd.shape[1]
    if write_start + rows > attend_len:
        raise ValueError(f"KV write [{write_start}, {write_start + rows}) past the attended rows [0, {attend_len})")
    k_cache, v_cache = _write_rows(k_nd, v_nd, k_cache_nd, v_cache_nd, layer, write_start)
    out = _attend(torch.from_dlpack(q_nd), k_cache[layer, :, :attend_len], v_cache[layer, :, :attend_len])
    return k_cache_nd, v_cache_nd, tvm.runtime.from_dlpack(out)


@tvm.register_global_func("fast_dllm.kv_write", override=True)
def _kv_write_packed(k_nd, v_nd, k_cache_nd, v_cache_nd, layer, start):
    """Write k/v at cache rows [start, + rows)."""
    _write_rows(k_nd, v_nd, k_cache_nd, v_cache_nd, layer, start)
    return k_cache_nd, v_cache_nd

# ─────────────────────────────────────────────────────────────────────────────


# Device-dispatch helpers. CUDA tensors use a zero-copy torch.from_dlpack view; other TVM devices
# (Metal) go through a .numpy() host copy (unified memory there).
_KDL_CUDA = 2


def _is_cuda(t: Any) -> bool:
    return t.device.dlpack_device_type() == _KDL_CUDA


def _tvm_to_torch(t: Any) -> torch.Tensor:
    if _is_cuda(t):
        return torch.from_dlpack(t)
    return torch.from_numpy(t.numpy())


# Static-shape dispatch. Metal artifacts carry fixed-row siblings of the dynamic functions
# (`model/metal.py` METAL_STATIC_FNS); with a static row count their GEMMs compile to much faster
# kernels. A call uses the sibling when the artifact has one for its row count, else the dynamic
# function, so CUDA/Jetson artifacts (no siblings) are unaffected. The opt-outs are read once per
# VM and function name; set them before the first call.
_STATIC_SUFFIX = {"forward_cached": "_b", "get_logits": "_m"}
_STATIC_OPT_OUT = {
    "forward_cached": "FAST_DLLM_DISABLE_B32",
    "get_logits": "FAST_DLLM_DISABLE_STATIC_LOGITS",
}
_static_fn_cache: dict[tuple[int, str, int], str] = {}


def static_fn_name(vm: Any, base: str, rows: int) -> str:
    """`base` + its static-shape suffix for `rows` (e.g. forward_cached_b32, get_logits_m8)
    when the VM has that function and its opt-out env var is not "1"; otherwise `base`."""
    key = (id(vm), base, int(rows))
    name = _static_fn_cache.get(key)
    if name is None:
        name = base
        if os.environ.get(_STATIC_OPT_OUT[base]) != "1":
            candidate = f"{base}{_STATIC_SUFFIX[base]}{int(rows)}"
            try:
                vm[candidate]
                name = candidate
            except Exception:  # noqa: BLE001 -- absent from this artifact
                pass
        _static_fn_cache[key] = name
    return name


# Device-side confidence. Metal artifacts carry get_confidence_m{rows} (`model/metal.py`), which
# reduces the logits to per-row argmax + max softmax probability on the GPU, so a step reads back
# 2 x rows values instead of [1, rows, vocab] fp32 logits. Without it (CUDA/Jetson artifacts, or
# FAST_DLLM_DISABLE_GPU_CONFIDENCE=1, read once per VM and row count) the logits are reduced in
# PyTorch on the host.
_confidence_fn_cache: dict[tuple[int, int], str | None] = {}


def confidence_fn_name(vm: Any, rows: int) -> str | None:
    key = (id(vm), int(rows))
    if key not in _confidence_fn_cache:
        name = None
        if os.environ.get("FAST_DLLM_DISABLE_GPU_CONFIDENCE") != "1":
            candidate = f"get_confidence_m{int(rows)}"
            try:
                vm[candidate]
                name = candidate
            except Exception:  # noqa: BLE001 -- absent from this artifact
                pass
        _confidence_fn_cache[key] = name
    return _confidence_fn_cache[key]


def _tvm_to_numpy(t: Any) -> np.ndarray:
    return t.numpy()


def shifted_confidence(vm: Any, params: list[Any], hidden: Any) -> tuple[np.ndarray, np.ndarray, int]:
    """Greedy token and confidence per position of `hidden` ([1, rows, hidden_size]).

    Returns:
        shifted_token_ids: [rows] int64 numpy   — argmax of the previous row's logits (row 0: its own)
        max_probs:         [rows] float32 numpy — matching max softmax probability
        last_token:        int — unshifted argmax of the last row (the token after the block)
    """
    rows = int(hidden.shape[1])
    conf_fn = confidence_fn_name(vm, rows)
    if conf_fn is not None:
        ids_tvm, maxp_tvm = vm[conf_fn](hidden, params)              # [1, rows] int32 / fp32
        ids = _tvm_to_numpy(ids_tvm)[0].astype(np.int64)
        maxp = _tvm_to_numpy(maxp_tvm)[0]
        shifted_ids = np.concatenate([ids[:1], ids[:-1]])
        shifted_maxp = np.concatenate([maxp[:1], maxp[:-1]])
        return shifted_ids, shifted_maxp, int(ids[-1])

    logits_tvm = vm[static_fn_name(vm, "get_logits", rows)](hidden, params)  # [1, rows, vocab]
    logits = _tvm_to_torch(logits_tvm)
    shifted = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
    shifted_token_ids = shifted.argmax(dim=-1)[0].cpu().numpy()      # [rows]
    max_probs = shifted.softmax(dim=-1).max(dim=-1).values[0].cpu().numpy()  # [rows]
    return shifted_token_ids, max_probs, int(logits[0, -1, :].argmax().item())


def selected_fn_names(vm: Any, *, block_size: int, small_block_size: int) -> dict[str, str]:
    """Which compiled function each generation call site dispatches to (for run metrics)."""
    return {
        "cached_forward_fn": static_fn_name(vm, "forward_cached", block_size),
        "append_forward_fn": static_fn_name(vm, "forward_cached", block_size),
        "logits_fn": static_fn_name(vm, "get_logits", block_size),
        "partial_logits_fn": static_fn_name(vm, "get_logits", small_block_size),
        "confidence_fn": confidence_fn_name(vm, block_size) or "host",
        "partial_confidence_fn": confidence_fn_name(vm, small_block_size) or "host",
    }


@dataclasses.dataclass(frozen=True)
class CacheSpec:
    """KV-cache dims of an artifact (from its kv_cache manifest, `export_utils.write_kv_cache_manifest`)."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: str


@dataclasses.dataclass
class PrefixCache:
    """Preallocated in-place KV cache for one sequence.

    k / v: TVM tensors [num_layers, 1, capacity, num_kv_heads, head_dim] (BSNH rows). Rows
    [0, prefix_len) hold the committed prefix and are never rewritten; compiled calls write the
    active rows at/after prefix_len (refinement, scoring) and `commit` makes them part of the prefix
    (block-append). Rows at/after the write frontier are uninitialized and never read.
    """

    k: Any
    v: Any
    prefix_len: int
    spec: CacheSpec

    @property
    def capacity(self) -> int:
        return int(self.k.shape[2])

    def commit(self, rows: int) -> None:
        if self.prefix_len + rows > self.capacity:
            raise ValueError(f"commit past capacity: {self.prefix_len} + {rows} > {self.capacity}")
        self.prefix_len += rows

    def ensure_capacity(self, rows: int, dev: Any) -> None:
        """Grow (reallocate, copy the committed rows) so that `rows` rows fit."""
        if rows <= self.capacity:
            return
        bigger = allocate_prefix_cache(self.spec, dev, rows)
        for old, new in ((self.k, bigger.k), (self.v, bigger.v)):
            torch.from_dlpack(new)[:, :, : self.prefix_len].copy_(torch.from_dlpack(old)[:, :, : self.prefix_len])
        self.k, self.v = bigger.k, bigger.v


def allocate_prefix_cache(spec: CacheSpec, dev: Any, rows: int, *, block_size: int = 32) -> PrefixCache:
    """An empty PrefixCache with capacity `rows` rounded up to a multiple of block_size."""
    capacity = -(-max(int(rows), 1) // block_size) * block_size
    shape = (spec.num_layers, 1, capacity, spec.num_kv_heads, spec.head_dim)
    return PrefixCache(
        k=tvm.runtime.empty(shape, spec.dtype, dev),
        v=tvm.runtime.empty(shape, spec.dtype, dev),
        prefix_len=0,
        spec=spec,
    )


def _check_write(cache: PrefixCache, write_start: int, attend_len: int) -> None:
    """Cached calls may only write at/after the committed prefix and inside the capacity."""
    if write_start < cache.prefix_len:
        raise ValueError(f"KV write at row {write_start} would overwrite the committed prefix [0, {cache.prefix_len})")
    if attend_len > cache.capacity:
        raise ValueError(f"cached call needs {attend_len} rows; cache capacity is {cache.capacity}")


def build_prefix_cache_from_ids(
    vm: Any,
    params: list[Any],
    dev: Any,
    reshape_fn: Any,
    prefix_ids: list[int],
    *,
    num_attention_heads: int,
    block_size: int,
    cache_spec: CacheSpec,
    capacity: int | None = None,
    chunked_prefill: bool = False,
) -> PrefixCache:
    """Allocate a PrefixCache (at least `capacity` rows) and fill rows [0, len(prefix_ids)) with
    the prefix's per-layer KV via build_prefix_cache.

    chunked_prefill: instead of one `build_prefix_cache` call with a dense block-causal bias
    over the whole prefix, build the cache one `block_size` chunk at a time through the bias-free
    `forward_cached` path decode already uses (block i attends fully to blocks 0..i, which is
    exactly the block-causal mask). The first chunk goes through the one-shot path (a single
    block, no prefix yet). Requires `len(prefix_ids) % block_size == 0`. Much faster for long
    prompts when the dense one-shot attention is not well optimized (e.g. the CUTLASS-off Jetson
    build); slower otherwise.

    num_attention_heads: unused (the prefill bias is shared across heads); kept for call-site
    compatibility.
    """
    prefix_len = len(prefix_ids)
    cache = allocate_prefix_cache(cache_spec, dev, max(prefix_len, capacity or 0), block_size=block_size)
    if prefix_len == 0:
        return cache
    if chunked_prefill and prefix_len > block_size:
        if prefix_len % block_size != 0:
            raise ValueError(
                f"chunked_prefill requires a block-aligned prefix; got {prefix_len} tokens "
                f"with block_size={block_size}."
            )
        _prefill(vm, params, dev, reshape_fn, prefix_ids[:block_size], cache, block_size=block_size)
        for start in range(block_size, prefix_len, block_size):
            _forward_cached(vm, params, dev, reshape_fn, prefix_ids[start : start + block_size], cache)
            cache.commit(block_size)
        return cache
    _prefill(vm, params, dev, reshape_fn, prefix_ids, cache, block_size=block_size)
    return cache


def _prefill(
    vm: Any, params: list[Any], dev: Any, reshape_fn: Any, prefix_ids: list[int], cache: PrefixCache, *, block_size: int
) -> None:
    """build_prefix_cache into an empty cache: write rows [0, len(prefix_ids)) and commit them."""
    if cache.prefix_len != 0:
        raise ValueError("build_prefix_cache writes from row 0; the cache already has a prefix")
    _check_write(cache, 0, len(prefix_ids))
    embedded = _embed(vm, params, dev, reshape_fn, prefix_ids)
    bias_np = build_prefix_block_attention_bias(prefix_len=0, block_len=len(prefix_ids), block_size=block_size)
    bias = tvm.runtime.tensor(bias_np, device=dev)
    vm["build_prefix_cache"](embedded, bias, cache.k, cache.v, params)
    cache.commit(len(prefix_ids))


def _embed(vm: Any, params: list[Any], dev: Any, reshape_fn: Any, ids: list[int]) -> Any:
    """vm["embed"] for `ids`, reshaped to [1, len(ids), hidden]."""
    tokens = tvm.runtime.tensor(np.asarray(ids, dtype="int32"), device=dev)
    embedded = vm["embed"](tokens, params)
    return reshape_fn(embedded, tvm.runtime.ShapeTuple([1, embedded.shape[0], embedded.shape[1]]))


def _forward_cached(
    vm: Any, params: list[Any], dev: Any, reshape_fn: Any, active_ids: list[int], cache: PrefixCache
) -> Any:
    """Compiled forward_cached for `active_ids` at the cache frontier (writes their K/V at rows
    [prefix_len, prefix_len + len); does not commit). Returns the final-norm hidden states."""
    _check_write(cache, cache.prefix_len, cache.prefix_len + len(active_ids))
    embedded = _embed(vm, params, dev, reshape_fn, active_ids)
    cache_offset = tvm.runtime.tensor(np.asarray(cache.prefix_len, dtype="int32"), device=dev)
    return vm[static_fn_name(vm, "forward_cached", len(active_ids))](
        embedded, cache.k, cache.v, cache_offset, tvm.runtime.ShapeTuple([cache.prefix_len]), params
    )


def score_active_block_cached_fast(
    vm: Any,
    params: list[Any],
    dev: Any,
    reshape_fn: Any,
    active_ids: list[int],
    prefix_cache: PrefixCache,
) -> tuple[np.ndarray, np.ndarray, int, PrefixCache]:
    """Full active-block scoring that commits the block to the cache: compiled forward_cached +
    `shifted_confidence`. Used at block-append; refinement steps use
    `score_active_block_refine`, which runs the same forward without committing.

    The lm_head GEMM runs in the artifact. The argmax/softmax/max reductions run on the device
    when the artifact has get_confidence_m{rows} (Metal); otherwise in PyTorch on a zero-copy
    DLPack view (CUDA), which is much faster than TVM's generic reduction kernels for
    [1, block_len, vocab_size] tensors there.

    Returns:
        shifted_token_ids:   [block_len] int64 numpy — token to unmask at each position
        max_probs:           [block_len] float32 numpy — confidence (threshold-comparable)
        block_append_token:  int — token predicted after the active block
        prefix_cache:        the same PrefixCache, with the active block committed
    """
    hidden = _forward_cached(vm, params, dev, reshape_fn, active_ids, prefix_cache)
    prefix_cache.commit(len(active_ids))

    shifted_token_ids, max_probs, block_append_token = shifted_confidence(vm, params, hidden)
    return shifted_token_ids, max_probs, block_append_token, prefix_cache


def score_active_block_refine(
    vm: Any,
    params: list[Any],
    dev: Any,
    reshape_fn: Any,
    active_ids: list[int],
    prefix_cache: PrefixCache,
) -> tuple[np.ndarray, np.ndarray]:
    """Full active-block refinement step: compiled forward_cached + `shifted_confidence`, not committed.

    The block's K/V are written at rows [prefix_len, prefix_len + block_len) -- where the partial
    per-layer forwards that may follow read them -- and the next call overwrites them.

    Returns:
        shifted_token_ids: [block_len] int64 numpy — token to unmask at each position
        max_probs:         [block_len] float32 numpy — confidence (threshold-comparable)
    """
    hidden = _forward_cached(vm, params, dev, reshape_fn, active_ids, prefix_cache)
    shifted_token_ids, max_probs, _ = shifted_confidence(vm, params, hidden)
    return shifted_token_ids, max_probs


def score_active_block_partial_perlayer(
    vm: Any,
    params: list[Any],
    dev: Any,
    reshape_fn: Any,
    small_block_ids: list[int],
    prefix_cache: PrefixCache,
    replace_position: int,
    *,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Score a small block with per-layer K/V update (HF-equivalent partial forward).

    Dispatches to vm["forward_block_partial_rp{replace_position}"], which overwrites the block's
    cache rows [replace_position, +sbs) with fresh K/V per layer before attending over prefix +
    the whole block — exact HF parity. The other block rows must hold the K/V of the last full
    refinement step (score_active_block_refine) of this block.

    Returns:
        shifted_token_ids:  [sbs] int64   — predictions for the sbs positions
        max_probs:          [sbs] float32  — softmax confidence (threshold-comparable)
    """
    write_start = prefix_cache.prefix_len + replace_position
    _check_write(prefix_cache, write_start, prefix_cache.prefix_len + block_size)
    embedded = _embed(vm, params, dev, reshape_fn, small_block_ids)
    cache_offset = tvm.runtime.tensor(np.asarray(write_start, dtype="int32"), device=dev)

    fn = vm[f"forward_block_partial_rp{replace_position}"]
    hidden = fn(
        embedded,
        prefix_cache.k,
        prefix_cache.v,
        cache_offset,
        tvm.runtime.ShapeTuple([prefix_cache.prefix_len]),
        params,
    )

    shifted_token_ids, max_probs, _ = shifted_confidence(vm, params, hidden)
    return shifted_token_ids, max_probs


def score_sequence_block(
    vm: Any,
    params: list[Any],
    dev: Any,
    reshape_fn: Any,
    sequence_ids: list[int],
    *,
    num_attention_heads: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw active-block logits and shifted block logits for a token sequence."""

    if len(sequence_ids) <= block_size:
        prefix_cache_ids: list[int] = []
        active_block_ids = list(sequence_ids)
    else:
        prefix_cache_len = ((len(sequence_ids) - 1) // block_size) * block_size
        prefix_cache_ids = sequence_ids[:prefix_cache_len]
        active_block_ids = sequence_ids[prefix_cache_len:]

    tokens = tvm.runtime.tensor(
        np.asarray(prefix_cache_ids + active_block_ids, dtype="int32"),
        device=dev,
    )
    attention_bias = build_prefix_block_attention_bias(
        prefix_len=len(prefix_cache_ids),
        block_len=len(active_block_ids),
        block_size=block_size,
    )
    bias = tvm.runtime.tensor(attention_bias, device=dev)
    embedded = vm["embed"](tokens, params)
    embedded = reshape_fn(
        embedded,
        tvm.runtime.ShapeTuple([1, embedded.shape[0], embedded.shape[1]]),
    )
    hidden = vm["forward_hidden"](embedded, bias, params)
    logits = vm["get_logits"](hidden, params).numpy()
    current_block_logits = logits[:, -len(active_block_ids) :, :]
    shifted = extract_shifted_block_logits(
        logits,
        prefix_len=len(prefix_cache_ids),
        block_len=len(active_block_ids),
    )
    return current_block_logits, shifted


def score_masked_continuation(
    vm: Any,
    params: list[Any],
    dev: Any,
    reshape_fn: Any,
    prefix_ids: list[int],
    continuation_ids: list[int],
    *,
    num_attention_heads: int,
    block_size: int,
    mask_id: int,
    cache_spec: CacheSpec,
    eos_token_id: int | None = None,
    cached_prefix: tuple[list[int], PrefixCache] | None = None,
    chunked_prefill: bool = False,
) -> tuple[np.ndarray, tuple[list[int], PrefixCache] | None]:
    """Score a continuation's tokens via one masked forward pass (lm-eval `loglikelihood`).

    Replaces every continuation position with `mask_id` before running the forward pass, then
    returns the logits at exactly those positions (one row per continuation token, in order),
    alongside the block-aligned prefix cache actually used (or `None` when nothing was
    cacheable). Masking is required: block-diffusion attention is bidirectional *within* a block,
    so feeding the real continuation tokens through unmasked would let each position see its own
    (and later) answer tokens. Masking the whole span in one pass generalizes the HF baseline's
    `FastDLLMBaselineRunner.loglikelihood` to continuations of any length.

    A `prefix_ids` longer than one block gets a KV cache for its block-aligned portion
    (`build_prefix_cache_from_ids`), and only the remaining tail (context remainder + masked
    continuation, padded to `block_size`) is scored via `forward_cached` — the same cached path
    generation uses, which keeps MMLU/GPQA's multi-thousand-token 5-shot contexts from
    materializing a dense attention matrix. A prefix no longer than one block falls back to
    `score_sequence_block`.

    `eos_token_id`: placed at the first padding position (matches the HF baseline's convention).
    `cached_prefix`: an optional `(aligned_prefix_ids, PrefixCache)` pair from a prior call,
    reused only on an exact match of the block-aligned prefix — MMLU/GPQA issue one request per
    answer choice with an identical context, so later choices skip the prefix build. Scoring writes
    the tail's K/V past the committed prefix and never commits, so the cache stays reusable.
    `cache_spec`: the artifact's KV-cache dims (for allocating the prefix cache).
    """
    if not continuation_ids:
        return np.empty((0, 0), dtype=np.float32), cached_prefix

    prefix_cache_len = (len(prefix_ids) // block_size) * block_size

    if prefix_cache_len == 0:
        sequence_ids = list(prefix_ids) + [mask_id] * len(continuation_ids)
        _, shifted = score_sequence_block(
            vm,
            params,
            dev,
            reshape_fn,
            sequence_ids,
            num_attention_heads=num_attention_heads,
            block_size=block_size,
        )
        return shifted[0, -len(continuation_ids) :, :], None  # nothing cacheable

    aligned_prefix_ids = prefix_ids[:prefix_cache_len]
    # Rows the scored tail needs: context remainder + continuation, padded up to block_size below.
    tail_rows = max(len(prefix_ids) - prefix_cache_len + len(continuation_ids), block_size)
    if cached_prefix is not None and cached_prefix[0] == aligned_prefix_ids:
        prefix_cache = cached_prefix[1]
        prefix_cache.ensure_capacity(prefix_cache_len + tail_rows, dev)
    else:
        prefix_cache = build_prefix_cache_from_ids(
            vm,
            params,
            dev,
            reshape_fn,
            aligned_prefix_ids,
            num_attention_heads=num_attention_heads,
            block_size=block_size,
            cache_spec=cache_spec,
            capacity=prefix_cache_len + tail_rows,
            chunked_prefill=chunked_prefill,
        )
    # active_ids = whatever of the context didn't fit in the block-aligned
    # cached prefix (0..block_size-1 real tokens), plus the masked continuation.
    active_real_ids = prefix_ids[prefix_cache.prefix_len :]
    active_ids = list(active_real_ids) + [mask_id] * len(continuation_ids)

    # Pad the active block up to block_size (first pad position = eos_token_id), so the
    # bidirectional within-block attention sees a full block_size window, as in generation and
    # the HF baseline. Skipped in the rare case active_ids already exceeds one block: the cached
    # path has no internal block-boundary masking, so padding past block_size would be unsafe.
    if len(active_ids) <= block_size:
        pad_len = block_size - len(active_ids)
        if pad_len:
            active_ids = active_ids + [mask_id] * pad_len
            if eos_token_id is not None:
                active_ids[len(active_real_ids) + len(continuation_ids)] = eos_token_id

    # Scoring never commits: the tail's K/V land past the prefix and are overwritten next time.
    hidden = _forward_cached(vm, params, dev, reshape_fn, active_ids, prefix_cache)
    logits_tvm = vm[static_fn_name(vm, "get_logits", hidden.shape[1])](hidden, params)
    logits = _tvm_to_torch(logits_tvm)  # [1, len(active_ids), vocab]
    # Shift-and-realign: the token AT position i is predicted from the logits AT position i-1.
    shifted = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
    # Bounded on both ends: padding (if any) follows the continuation, so an
    # open-ended slice would leak padding-position logits into the result.
    continuation_end = len(active_real_ids) + len(continuation_ids)
    logits_out = shifted[0, len(active_real_ids) : continuation_end, :].cpu().numpy()
    return logits_out, (aligned_prefix_ids, prefix_cache)


def relax_greedy_generate(
    vm: Any,
    params: list[Any],
    dev: Any,
    reshape_fn: Any,
    tokenizer: Any,
    prompt_token_ids: list[int],
    *,
    num_attention_heads: int,
    max_new_tokens: int,
    block_size: int,
    small_block_size: int,
    threshold: float,
    temperature: float,
    top_p: float,
    use_block_cache: bool,
    mask_id: int,
    stop_token: int | None,
    cache_spec: CacheSpec,
    timing: dict[str, Any] | None = None,
    chunked_prefill: bool = False,
) -> tuple[list[int], dict[str, Any]]:
    """Fast-dLLM v2 block-diffusion generation (greedy, threshold-parallel unmasking).

    Mirrors the HF reference generator: each `block_size` block is refined in `small_block_size`
    windows; each refinement step unmasks every masked position whose confidence exceeds
    `threshold` (at least one). The first block (no prefix yet) uses the stateless scorer;
    later blocks use the prefix KV cache, and — with `use_block_cache` — partial per-layer
    re-forwards of just the current small block.

    The prefix KV cache is one preallocated buffer sized for the whole generation; block-append
    commits the finished block in place (`PrefixCache.commit`).

    cache_spec: the artifact's KV-cache dims.
    chunked_prefill: build the prefix KV cache block-by-block; see `build_prefix_cache_from_ids`.
    timing: when a dict, `prefill_latency_s` is recorded in it.
    """

    if temperature != 0.0:
        raise NotImplementedError("Fast-dLLM Relax runner currently supports only temperature=0.")
    if top_p <= 0:
        raise ValueError("top_p must be positive.")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if block_size <= 0 or small_block_size <= 0:
        raise ValueError("block_size and small_block_size must be positive.")

    _call_start = time.perf_counter()
    _prefill_recorded = False

    generated = list(prompt_token_ids)
    original_prompt_len = len(generated)
    unmask_updates = 0
    block_recomputes = 0
    partial_recomputes = 0
    block_appends = 0
    prefix_cache: PrefixCache | None = None
    # Whether the active block's cache rows hold the K/V of a full refinement step of this block
    # (the partial per-layer forwards update them window by window).
    block_kv_valid = False
    # Every row the generation can write: the prompt plus max_new_tokens, block-aligned, plus the
    # active block that may extend past the token limit.
    cache_capacity = -(-(original_prompt_len + max_new_tokens) // block_size) * block_size + block_size

    def _has_completed_stop(sequence_ids: list[int], prompt_length: int) -> bool:
        if stop_token is None:
            return False
        suffix = sequence_ids[prompt_length:]
        if stop_token not in suffix:
            return False
        stop_index = suffix.index(stop_token)
        return mask_id not in suffix[:stop_index]

    def _make_continuation(tokens: list[int]) -> list[int]:
        # Trim at the first mask token: can appear when max_new_tokens cuts mid-block
        # refinement before the active block is fully unmasked.
        raw = tokens[original_prompt_len : original_prompt_len + max_new_tokens]
        first_mask = next((i for i, t in enumerate(raw) if t == mask_id), len(raw))
        return raw[:first_mask]

    def _stats() -> dict[str, int]:
        return {
            "unmask_updates": unmask_updates,
            "block_recomputes": block_recomputes,
            "partial_recomputes": partial_recomputes,
            "block_appends": block_appends,
        }

    if original_prompt_len % block_size == 0 and original_prompt_len > 0 and max_new_tokens > 0:
        # Prompt ending on a block boundary: like HF's block-append after prefill, the first decode
        # block starts with the token predicted from the last prompt token (not from a mask's own
        # logits). Prefill all but the last prompt block, then run that block through forward_cached:
        # it commits the block and yields the token, so this costs no more than prefilling the whole
        # prompt. A 32-token prompt prefills nothing; forward_cached at P=0 is then plain
        # single-block (bidirectional) attention -- unlike HF, which forwards only the masks there.
        last_block_start = original_prompt_len - block_size
        prefix_cache = build_prefix_cache_from_ids(
            vm,
            params,
            dev,
            reshape_fn,
            generated[:last_block_start],
            num_attention_heads=num_attention_heads,
            block_size=block_size,
            cache_spec=cache_spec,
            capacity=cache_capacity,
            chunked_prefill=chunked_prefill,
        )
        _, _, next_token_id, prefix_cache = score_active_block_cached_fast(
            vm, params, dev, reshape_fn, generated[last_block_start:], prefix_cache
        )
        block_recomputes += 1
        block_appends += 1
        generated.append(next_token_id)
        if stop_token is not None and next_token_id == int(stop_token):
            if timing is not None:
                timing["prefill_latency_s"] = time.perf_counter() - _call_start
            return _make_continuation(generated), _stats()

    while len(generated) - original_prompt_len < max_new_tokens:
        prompt_length = len(generated)
        # In [1, block_size]; a block-aligned prompt got its first token above, and after every
        # block-append the sequence is one token past a block boundary.
        mask_padding = block_size - (prompt_length % block_size)
        x_t = generated + [mask_id] * mask_padding
        active_start = len(x_t) - block_size

        # Decide whether to use the cached path and ensure the prefix cache is current.
        if active_start == 0:
            # First block — no prefix exists yet; use the stateless scorer.
            use_cache = False
        else:
            use_cache = True
            current_prefix_len = prefix_cache.prefix_len if prefix_cache is not None else 0
            if active_start != current_prefix_len:
                # Build (or rebuild) prefix cache from the completed prefix tokens.
                prefix_cache = build_prefix_cache_from_ids(
                    vm,
                    params,
                    dev,
                    reshape_fn,
                    x_t[:active_start],
                    num_attention_heads=num_attention_heads,
                    block_size=block_size,
                    cache_spec=cache_spec,
                    capacity=cache_capacity,
                    chunked_prefill=chunked_prefill,
                )
                block_kv_valid = False

        if not _prefill_recorded:
            # Mirrors HF's own prefill boundary: one forward call (here, one prefix-cache build,
            # or none at all when active_start == 0) made before any block-unmasking work starts.
            if timing is not None:
                timing["prefill_latency_s"] = time.perf_counter() - _call_start
            _prefill_recorded = True

        for small_block_start in range(0, block_size, small_block_size):
            small_block_end = min(block_size, small_block_start + small_block_size)
            while True:
                block_slice = x_t[active_start + small_block_start : active_start + small_block_end]
                local_mask_positions = [index for index, tok in enumerate(block_slice) if tok == mask_id]
                if not local_mask_positions:
                    break
                if _has_completed_stop(x_t, prompt_length):
                    return _make_continuation(x_t), _stats()

                if use_cache:
                    # Use partial forward when block cache is populated AND the first position
                    # of the current small block is already a real token (mirrors the HF
                    # reference generator's condition).
                    use_partial = (
                        use_block_cache
                        and block_kv_valid
                        and x_t[active_start + small_block_start] != mask_id
                    )
                    if use_partial:
                        # Partial forward: sbs tokens, per-layer K update of the block's cache rows.
                        shifted_ids, probs_arr = score_active_block_partial_perlayer(
                            vm,
                            params,
                            dev,
                            reshape_fn,
                            x_t[active_start + small_block_start : active_start + small_block_end],
                            prefix_cache,  # type: ignore[arg-type]
                            replace_position=small_block_start,
                            block_size=block_size,
                        )
                    else:
                        # Full block forward: block_size tokens. Writes the block's K/V past the
                        # prefix without committing (only block-append below commits).
                        shifted_ids, probs_arr = score_active_block_refine(
                            vm,
                            params,
                            dev,
                            reshape_fn,
                            x_t[active_start:],
                            prefix_cache,  # type: ignore[arg-type]
                        )
                        block_kv_valid = use_block_cache
                    block_recomputes += 1
                    if use_partial:
                        partial_recomputes += 1
                    block_ids = shifted_ids if use_partial else shifted_ids[small_block_start:small_block_end]
                    block_probs = probs_arr if use_partial else probs_arr[small_block_start:small_block_end]
                    unmask_rows = [row for row in local_mask_positions if block_probs[row] > threshold]
                    if not unmask_rows:
                        best_row = local_mask_positions[
                            int(np.argmax([block_probs[r] for r in local_mask_positions]))
                        ]
                        unmask_rows = [best_row]
                    _base = active_start + small_block_start
                    for row in unmask_rows:
                        x_t[_base + row] = int(block_ids[row])
                        unmask_updates += 1
                else:
                    _, shifted_logits = score_sequence_block(
                        vm,
                        params,
                        dev,
                        reshape_fn,
                        x_t,
                        num_attention_heads=num_attention_heads,
                        block_size=block_size,
                    )
                    # Stateless path: every call allocates fresh buffers for the whole sequence
                    # and reuses nothing, so return them to the pool right away.
                    release_relax_memory_pool()
                    block_recomputes += 1
                    small_logits = shifted_logits[:, small_block_start:small_block_end, :][0]
                    chosen_ids = small_logits.argmax(axis=-1)
                    chosen_probs = np.empty(small_logits.shape[0], dtype=np.float64)
                    for _row in range(small_logits.shape[0]):
                        _r = small_logits[_row].astype(np.float64)
                        _e = np.exp(_r - _r.max())
                        chosen_probs[_row] = _e[chosen_ids[_row]] / _e.sum()
                    gated_probs = np.full_like(chosen_probs, -np.inf, dtype=np.float64)
                    for row in local_mask_positions:
                        gated_probs[row] = chosen_probs[row]
                    unmask_rows = [row for row in local_mask_positions if gated_probs[row] > threshold]
                    if not unmask_rows:
                        best_row = int(np.argmax(gated_probs))
                        unmask_rows = [best_row]
                    for row in unmask_rows:
                        token_id = int(chosen_ids[row])
                        x_t[active_start + small_block_start + row] = token_id
                        unmask_updates += 1

                if len(x_t) - original_prompt_len >= max_new_tokens:
                    break

            if len(x_t) - original_prompt_len >= max_new_tokens:
                break

        generated = x_t
        if len(generated) - original_prompt_len >= max_new_tokens:
            break
        if _has_completed_stop(generated, original_prompt_len):
            break

        # Block-append: predict the token after the completed block and extend prefix cache.
        if use_cache:
            _, _, next_token_id, prefix_cache = score_active_block_cached_fast(
                vm,
                params,
                dev,
                reshape_fn,
                generated[active_start:],
                prefix_cache,  # type: ignore[arg-type]
            )
            block_kv_valid = False  # the next active block starts unrefined
        else:
            current_block_logits, _ = score_sequence_block(
                vm,
                params,
                dev,
                reshape_fn,
                generated,
                num_attention_heads=num_attention_heads,
                block_size=block_size,
            )
            next_token_id = int(current_block_logits[0, -1, :].argmax())
            release_relax_memory_pool()  # see the matching call above
        block_recomputes += 1
        generated.append(next_token_id)
        block_appends += 1
        if stop_token is not None and next_token_id == int(stop_token):
            break

    return _make_continuation(generated), _stats()
