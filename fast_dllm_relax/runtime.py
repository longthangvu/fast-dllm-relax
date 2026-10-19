"""Runtime helpers for Fast-dLLM Relax prompt scoring and generation."""

from __future__ import annotations

import dataclasses
import math as _math
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as _F
import tvm

from .common import release_relax_memory_pool
from .model.cuda import build_prefix_block_attention_bias, extract_shifted_block_logits

# ── bf16 attention packed function ────────────────────────────────────────────
# Registered at import time so it's available before any VM is created. The CUDA model's
# forward_cached / forward_block_partial_rp* call it via
# rx.call_pure_packed("fast_dllm.flash_attn_bf16"), once per layer, as a synchronous VM-to-Python
# callback. Inputs/outputs are BSNH TVM tensors; GQA is handled natively (no K/V repeat).
_FLASH_SCALE = 1.0 / _math.sqrt(128)  # head_dim=128

try:
    import flash_attn as _flash_attn

    @tvm.register_global_func("fast_dllm.flash_attn_bf16", override=True)
    def _flash_attn_bf16_packed(q_nd, k_nd, v_nd):
        q = torch.from_dlpack(q_nd)
        k = torch.from_dlpack(k_nd)
        v = torch.from_dlpack(v_nd)
        out = _flash_attn.flash_attn_func(q, k, v, softmax_scale=_FLASH_SCALE, causal=False)
        return tvm.runtime.from_dlpack(out.contiguous())

except ImportError:
    # Fallback: PyTorch SDPA dispatches to its native FlashAttention-2 kernel when attn_mask=None.
    # enable_gqa=True attends on the native num_kv_heads tensors; bitwise-identical to an explicit
    # repeat-then-attend at this model's shapes/dtype.

    @tvm.register_global_func("fast_dllm.flash_attn_bf16", override=True)
    def _flash_attn_bf16_packed(q_nd, k_nd, v_nd):
        q = torch.from_dlpack(q_nd)  # BSNH
        k = torch.from_dlpack(k_nd)
        v = torch.from_dlpack(v_nd)
        # SDPA expects BHSD
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = _F.scaled_dot_product_attention(
            q, k, v, scale=_FLASH_SCALE, is_causal=False, enable_gqa=True
        )
        out = out.transpose(1, 2).contiguous()  # BHSD → BSNH
        return tvm.runtime.from_dlpack(out)

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


@dataclasses.dataclass
class BlockCache:
    """Per-layer active-block KV cache for within-block refinement.

    block_k / block_v: TVM tensors with shape
        [num_layers, 1, num_kv_heads, block_size, head_dim]
    (kept as TVM tensors so they can be passed directly to compiled functions
    without DLPack round-trips; updated by the per-layer compiled forward.)
    """

    block_k: Any
    block_v: Any
    block_size: int
    small_block_size: int


@dataclasses.dataclass
class PrefixCache:
    """Per-layer prefix KV cache between generation steps.

    cached_k / cached_v: TVM tensors with shape
        [num_layers, 1, num_kv_heads, prefix_len, head_dim]
    """

    cached_k: Any
    cached_v: Any
    prefix_len: int


def build_prefix_cache_from_ids(
    vm: Any,
    params: list[Any],
    dev: Any,
    reshape_fn: Any,
    prefix_ids: list[int],
    *,
    num_attention_heads: int,
    block_size: int,
    chunked_prefill: bool = False,
) -> PrefixCache:
    """Embed completed prefix blocks and run build_prefix_cache to get per-layer KV.

    chunked_prefill: instead of one `build_prefix_cache` call with a dense block-causal bias
    over the whole prefix, build the cache one `block_size` chunk at a time through the bias-free
    `forward_cached` path decode already uses (block i attends fully to blocks 0..i, which is
    exactly the block-causal mask). The first chunk goes through the one-shot path (a single
    block, no prefix yet) so the cache's shape/dtype come from the artifact itself. Requires
    `len(prefix_ids) % block_size == 0`. Much faster for long prompts when the dense one-shot
    attention is not well optimized (e.g. the CUTLASS-off Jetson build); slower otherwise.
    """
    prefix_len = len(prefix_ids)
    if chunked_prefill and prefix_len > block_size:
        if prefix_len % block_size != 0:
            raise ValueError(
                f"chunked_prefill requires a block-aligned prefix; got {prefix_len} tokens "
                f"with block_size={block_size}."
            )
        cache = build_prefix_cache_from_ids(
            vm, params, dev, reshape_fn, prefix_ids[:block_size],
            num_attention_heads=num_attention_heads, block_size=block_size,
        )
        for start in range(block_size, prefix_len, block_size):
            tokens = tvm.runtime.tensor(
                np.asarray(prefix_ids[start : start + block_size], dtype="int32"), device=dev
            )
            embedded = vm["embed"](tokens, params)
            embedded = reshape_fn(
                embedded,
                tvm.runtime.ShapeTuple([1, embedded.shape[0], embedded.shape[1]]),
            )
            cache_offset = tvm.runtime.tensor(np.asarray(start, dtype="int32"), device=dev)
            _, new_k, new_v = vm["forward_cached"](
                embedded, cache.cached_k, cache.cached_v, cache_offset, params
            )
            cache = PrefixCache(cached_k=new_k, cached_v=new_v, prefix_len=start + block_size)
            # The pooled allocator only reuses exact-size buffers, so each growing cache would
            # otherwise leave a never-reused buffer behind.
            release_relax_memory_pool()
        return cache
    tokens = tvm.runtime.tensor(np.asarray(prefix_ids, dtype="int32"), device=dev)
    embedded = vm["embed"](tokens, params)
    embedded = reshape_fn(
        embedded,
        tvm.runtime.ShapeTuple([1, embedded.shape[0], embedded.shape[1]]),
    )
    bias_np = build_prefix_block_attention_bias(
        prefix_len=0,
        block_len=prefix_len,
        num_attention_heads=num_attention_heads,
        block_size=block_size,
    )
    bias = tvm.runtime.tensor(bias_np.astype("float32"), device=dev)
    cached_k, cached_v = vm["build_prefix_cache"](embedded, bias, params)
    return PrefixCache(cached_k=cached_k, cached_v=cached_v, prefix_len=prefix_len)


def score_active_block_cached_fast(
    vm: Any,
    params: list[Any],
    dev: Any,
    reshape_fn: Any,
    active_ids: list[int],
    prefix_cache: PrefixCache,
) -> tuple[np.ndarray, np.ndarray, int, PrefixCache]:
    """Full active-block scoring: compiled forward_cached + get_logits, PyTorch reductions.

    The lm_head GEMM runs in the artifact (get_logits); argmax/softmax/max run in PyTorch on a
    zero-copy DLPack view, which is much faster than TVM's generic reduction kernels for
    [1, block_len, vocab_size] tensors.

    Returns:
        shifted_token_ids:   [block_len] int64 numpy — token to unmask at each position
        max_probs:           [block_len] float32 numpy — confidence (threshold-comparable)
        block_append_token:  int — token predicted after the active block
        new_prefix_cache:    updated PrefixCache (KV extended to include active block)
    """
    block_len = len(active_ids)
    tokens = tvm.runtime.tensor(np.asarray(active_ids, dtype="int32"), device=dev)
    embedded = vm["embed"](tokens, params)
    embedded = reshape_fn(
        embedded,
        tvm.runtime.ShapeTuple([1, embedded.shape[0], embedded.shape[1]]),
    )
    cache_offset = tvm.runtime.tensor(
        np.asarray(prefix_cache.prefix_len, dtype="int32"), device=dev
    )
    hidden, new_cached_k, new_cached_v = vm["forward_cached"](
        embedded, prefix_cache.cached_k, prefix_cache.cached_v, cache_offset, params
    )

    logits_tvm = vm["get_logits"](hidden, params)                    # [1, block_len, vocab]
    logits = _tvm_to_torch(logits_tvm)
    shifted = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
    shifted_token_ids = shifted.argmax(dim=-1)[0].cpu().numpy()      # [block_len]
    max_probs = shifted.softmax(dim=-1).max(dim=-1).values[0].cpu().numpy()  # [block_len]
    block_append_token = int(logits[0, -1, :].argmax().item())       # scalar

    new_prefix_cache = PrefixCache(
        cached_k=new_cached_k,
        cached_v=new_cached_v,
        prefix_len=prefix_cache.prefix_len + block_len,
    )
    return shifted_token_ids, max_probs, block_append_token, new_prefix_cache


def extract_block_kv(
    new_cached_k: Any,
    new_cached_v: Any,
    prefix_len: int,
    block_size: int,
    small_block_size: int,
) -> BlockCache:
    """Extract the active-block KV from the extended prefix cache returned by forward_cached.

    forward_cached returns new_cached_k of shape
      [num_layers, 1, num_kv_heads, prefix_len + block_size, head_dim].
    The block portion is the last block_size positions.

    Returns TVM tensors so they can be passed directly to compiled partial-forward
    functions without an extra DLPack round-trip per call.
    """
    if _is_cuda(new_cached_k):
        ck = torch.from_dlpack(new_cached_k)  # zero-copy CUDA view
        cv = torch.from_dlpack(new_cached_v)
        bk_pt = ck[:, :, :, prefix_len : prefix_len + block_size, :].contiguous()
        bv_pt = cv[:, :, :, prefix_len : prefix_len + block_size, :].contiguous()
        return BlockCache(
            block_k=tvm.runtime.from_dlpack(bk_pt),
            block_v=tvm.runtime.from_dlpack(bv_pt),
            block_size=block_size,
            small_block_size=small_block_size,
        )
    # Non-CUDA: host-copy slice, then back onto the cache's own device.
    dev = new_cached_k.device
    bk_np = np.ascontiguousarray(new_cached_k.numpy()[:, :, :, prefix_len : prefix_len + block_size, :])
    bv_np = np.ascontiguousarray(new_cached_v.numpy()[:, :, :, prefix_len : prefix_len + block_size, :])
    return BlockCache(
        block_k=tvm.runtime.tensor(bk_np, device=dev),
        block_v=tvm.runtime.tensor(bv_np, device=dev),
        block_size=block_size,
        small_block_size=small_block_size,
    )


def score_active_block_partial_perlayer(
    vm: Any,
    params: list[Any],
    dev: Any,
    reshape_fn: Any,
    small_block_ids: list[int],
    prefix_cache: PrefixCache,
    block_cache: BlockCache,
    replace_position: int,
) -> tuple[np.ndarray, np.ndarray, BlockCache]:
    """Score a small block with per-layer K/V update (HF-equivalent partial forward).

    Dispatches to vm["forward_block_partial_rp{replace_position}"], which splices fresh K/V
    into block_k at replace_position per layer before attending — exact HF parity.
    No Python-side torch.cat needed; block_k/v are passed as TVM tensors directly.

    Returns:
        shifted_token_ids:  [sbs] int64   — predictions for the sbs positions
        max_probs:          [sbs] float32  — softmax confidence (threshold-comparable)
        new_block_cache:    BlockCache with updated block_k/v from the compiled function
    """
    tokens = tvm.runtime.tensor(np.asarray(small_block_ids, dtype="int32"), device=dev)
    embedded = vm["embed"](tokens, params)
    embedded = reshape_fn(
        embedded,
        tvm.runtime.ShapeTuple([1, embedded.shape[0], embedded.shape[1]]),
    )
    cache_offset = tvm.runtime.tensor(
        np.asarray(prefix_cache.prefix_len + replace_position, dtype="int32"), device=dev
    )

    fn = vm[f"forward_block_partial_rp{replace_position}"]
    hidden, _new_active_k, new_bk_tvm, new_bv_tvm = fn(
        embedded,
        prefix_cache.cached_k,
        prefix_cache.cached_v,
        block_cache.block_k,
        block_cache.block_v,
        cache_offset,
        params,
    )

    # get_logits + DLPack + PyTorch reductions (same pattern as score_active_block_cached_fast)
    logits_tvm = vm["get_logits"](hidden, params)
    logits = _tvm_to_torch(logits_tvm)                              # [1, sbs, vocab]
    shifted = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
    shifted_token_ids = shifted.argmax(dim=-1)[0].cpu().numpy()     # [sbs]
    max_probs = shifted.softmax(dim=-1).max(dim=-1).values[0].cpu().numpy()  # [sbs]

    new_block_cache = BlockCache(
        block_k=new_bk_tvm,
        block_v=new_bv_tvm,
        block_size=block_cache.block_size,
        small_block_size=block_cache.small_block_size,
    )
    return shifted_token_ids, max_probs, new_block_cache


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
        num_attention_heads=num_attention_heads,
        block_size=block_size,
    )
    bias = tvm.runtime.tensor(attention_bias.astype("float32"), device=dev)
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
    answer choice with an identical context, so later choices skip the prefix build.
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
    if cached_prefix is not None and cached_prefix[0] == aligned_prefix_ids:
        prefix_cache = cached_prefix[1]
    else:
        prefix_cache = build_prefix_cache_from_ids(
            vm,
            params,
            dev,
            reshape_fn,
            aligned_prefix_ids,
            num_attention_heads=num_attention_heads,
            block_size=block_size,
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

    tokens = tvm.runtime.tensor(np.asarray(active_ids, dtype="int32"), device=dev)
    embedded = vm["embed"](tokens, params)
    embedded = reshape_fn(
        embedded,
        tvm.runtime.ShapeTuple([1, embedded.shape[0], embedded.shape[1]]),
    )
    cache_offset = tvm.runtime.tensor(np.asarray(prefix_cache.prefix_len, dtype="int32"), device=dev)
    hidden, _new_cached_k, _new_cached_v = vm["forward_cached"](
        embedded, prefix_cache.cached_k, prefix_cache.cached_v, cache_offset, params
    )
    logits_tvm = vm["get_logits"](hidden, params)
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
    timing: dict[str, Any] | None = None,
    chunked_prefill: bool = False,
) -> tuple[list[int], dict[str, Any]]:
    """Fast-dLLM v2 block-diffusion generation (greedy, threshold-parallel unmasking).

    Mirrors the HF reference generator: each `block_size` block is refined in `small_block_size`
    windows; each refinement step unmasks every masked position whose confidence exceeds
    `threshold` (at least one). The first block (no prefix yet) uses the stateless scorer;
    later blocks use the prefix KV cache, and — with `use_block_cache` — partial per-layer
    re-forwards of just the current small block.

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
    block_cache: BlockCache | None = None

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

    while len(generated) - original_prompt_len < max_new_tokens:
        prompt_length = len(generated)
        mask_padding = block_size - (prompt_length % block_size)
        if mask_padding == 0:
            mask_padding = block_size
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
                    chunked_prefill=chunked_prefill,
                )
                block_cache = None  # prefix changed; invalidate block cache
                release_relax_memory_pool()

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
                    return _make_continuation(x_t), {
                        "unmask_updates": unmask_updates,
                        "block_recomputes": block_recomputes,
                        "partial_recomputes": partial_recomputes,
                        "block_appends": block_appends,
                    }

                if use_cache:
                    # Use partial forward when block cache is populated AND the first position
                    # of the current small block is already a real token (mirrors the HF
                    # reference generator's condition).
                    use_partial = (
                        use_block_cache
                        and block_cache is not None
                        and x_t[active_start + small_block_start] != mask_id
                    )
                    if use_partial:
                        # Partial forward: sbs tokens, per-layer K update inside compiled fn.
                        shifted_ids, probs_arr, block_cache = score_active_block_partial_perlayer(
                            vm,
                            params,
                            dev,
                            reshape_fn,
                            x_t[active_start + small_block_start : active_start + small_block_end],
                            prefix_cache,  # type: ignore[arg-type]
                            block_cache,  # type: ignore[arg-type]
                            replace_position=small_block_start,
                        )
                    else:
                        # Full block forward: block_size tokens, builds/updates block cache.
                        shifted_ids, probs_arr, _, new_prefix_cache = score_active_block_cached_fast(
                            vm,
                            params,
                            dev,
                            reshape_fn,
                            x_t[active_start:],
                            prefix_cache,  # type: ignore[arg-type]
                        )
                        if use_block_cache:
                            block_cache = extract_block_kv(
                                new_prefix_cache.cached_k,
                                new_prefix_cache.cached_v,
                                prefix_len=prefix_cache.prefix_len,  # type: ignore[union-attr]
                                block_size=block_size,
                                small_block_size=small_block_size,
                            )
                        # Do NOT advance prefix_cache here — keep prefix stable during refinement.
                        # (new_prefix_cache is only committed at block-append below.)
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
            block_cache = None  # reset for the next active block
            release_relax_memory_pool()
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

    return _make_continuation(generated), {
        "unmask_updates": unmask_updates,
        "block_recomputes": block_recomputes,
        "partial_recomputes": partial_recomputes,
        "block_appends": block_appends,
    }
