import dataclasses
import math

import numpy as np

import tvm
from tvm import relax as rx
from tvm.relax import register_pipeline
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.relax.frontend.nn.core import wrap_nested
from tvm.s_tir import dlight


@dataclasses.dataclass(frozen=True)
class FastDLLMRelaxConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    context_window_size: int
    tie_word_embeddings: bool
    bd_size: int
    pad_token_id: int


@register_pipeline("opt_fast_dllm_llm_cublas")
def opt_fast_dllm_llm_cublas(
    ext_mods: list[nn.ExternModule] | None = None,
    use_cutlass: bool = True,
    verbose: bool = True,
):
    """Production CUDA pipeline: cuBLAS for projection GEMMs, CUTLASS for batched matmuls, dlight.

    Requires TVM built with USE_CUBLAS=ON (and USE_CUTLASS=ON when use_cutlass=True).

    partition_for_cublas must run on the original Relax IR (before TIR lowering). LambdaLift is
    inserted before VMShapeLower because cuBLAS emits nested composite functions that
    VMShapeLower requires to be in global scope.

    use_cutlass: whether to run the CUTLASS FuseOpsByPattern step. CUTLASS's batched-GEMM pattern
    matches the model's manual 4D Q@K^T / probs@V matmuls (cuBLAS's pattern rejects
    batched-x-batched matmuls). CUTLASS's RunCodegen compiles its generated CUDA host wrapper with
    a plain, non-cross-compile-aware nvcc invocation, so the object is always built for the local
    host; set False for cross-compiled (Jetson) exports. The affected matmuls then fall back to
    dlight scheduling.

    verbose: print each pass name and its wall time (compilation takes minutes).
    """
    ext_mods = ext_mods or []

    @tvm.transform.module_pass(opt_level=0)
    def pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        from tvm.relax.backend.pattern_registry import get_patterns_with_prefix  # noqa: PLC0415
        import tvm.relax.backend.cuda.cublas  # noqa: F401, PLC0415  # registers cuBLAS patterns
        import tvm.relax.backend.cuda.cutlass  # noqa: F401, PLC0415  # registers CUTLASS patterns
        # Partition every hot-path function, including build_prefix_cache: dlight-scheduled bf16
        # GEMMs diverge from HF far more than cuBLAS's do, and the prefix cache's precision
        # carries into every later generation step even though it is built once per prompt.
        _cublas_hot_fns = [
            "forward_hidden", "get_logits",
            "forward_cached",
            "forward_block_partial_rp0", "forward_block_partial_rp8",
            "forward_block_partial_rp16", "forward_block_partial_rp24",
            "build_prefix_cache",
        ]
        # cuBLAS MUST be partitioned first. CUTLASS's FuseOpsByPattern also matches a generic
        # dense/matmul pattern, i.e. the same 2D-weight projection GEMMs (qkv/o_proj/gate_up/
        # down_proj/get_logits) cuBLAS wants; running CUTLASS first lets it claim them and cuBLAS
        # silently partitions nothing (scripts/export_cuda.py asserts against that). cuBLAS's own
        # pattern rejects batched-x-batched matmuls, so running it first only claims the
        # projections, leaving the batched attention matmuls to CUTLASS.
        mod = rx.transform.FuseOpsByPattern(
            get_patterns_with_prefix("cublas"),
            bind_constants=False,
            annotate_codegen=True,
            entry_functions=_cublas_hot_fns,
        )(mod)
        if use_cutlass:
            mod = rx.transform.FuseOpsByPattern(
                get_patterns_with_prefix("cutlass"),
                bind_constants=False,
                annotate_codegen=True,
                entry_functions=_cublas_hot_fns,
            )(mod)
        # RunCodegen must always run: it is what compiles the cuBLAS-tagged functions too. The
        # "cutlass" key (and its nvcc config) is only added when use_cutlass is True, so the
        # cross-compile-unsafe CUTLASS nvcc build is never invoked otherwise.
        target_options = {"cublas": {}}
        if use_cutlass:
            import tempfile as _tempfile  # noqa: PLC0415
            _cutlass_tmp = _tempfile.mkdtemp(prefix="cutlass_build_")
            target_options["cutlass"] = {"sm": 80, "find_first_valid": False, "tmp_dir": _cutlass_tmp}
        mod = rx.transform.RunCodegen(target_options)(mod)  # sm80 kernels also run on sm86/sm89
        import time as _time  # noqa: PLC0415
        # dlight.gpu.Matmul() hangs on the batched mixed-precision (fp16/bf16 -> fp32) attention
        # matmuls. With CUTLASS on, CUTLASS consumes those before scheduling, so Matmul() never sees
        # them; with use_cutlass=False (Jetson) drop Matmul() and let them fall through to
        # dlight.gpu.Fallback() (slower per kernel, but terminates).
        _matmul_rules = [dlight.gpu.Matmul()] if use_cutlass else []
        _passes = [
            ("FuseTransposeMatmul",      rx.transform.FuseTransposeMatmul()),
            ("LegalizeOps",              rx.transform.LegalizeOps()),
            ("AnnotateTIROpPattern",     rx.transform.AnnotateTIROpPattern()),
            ("FoldConstant",             rx.transform.FoldConstant()),
            ("FuseOps",                  rx.transform.FuseOps()),
            ("FuseTIR",                  rx.transform.FuseTIR()),
            ("DeadCodeElimination",      rx.transform.DeadCodeElimination()),
            ("ApplyDefaultSchedule",     dlight.ApplyDefaultSchedule(
                *_matmul_rules,
                dlight.gpu.GEMV(),
                dlight.gpu.Reduction(),
                dlight.gpu.GeneralReduction(),
                dlight.gpu.Fallback(),
            )),
            ("RewriteDataflowReshape",   rx.transform.RewriteDataflowReshape()),
            ("ToNonDataflow",            rx.transform.ToNonDataflow()),
            ("RemovePurityChecking",     rx.transform.RemovePurityChecking()),
            ("CallTIRRewrite",           rx.transform.CallTIRRewrite()),
            ("StaticPlanBlockMemory",    rx.transform.StaticPlanBlockMemory()),
            ("RewriteCUDAGraph",         rx.transform.RewriteCUDAGraph()),
            ("LowerAllocTensor",         rx.transform.LowerAllocTensor()),
            ("KillAfterLastUse",         rx.transform.KillAfterLastUse()),
            ("LowerRuntimeBuiltin",      rx.transform.LowerRuntimeBuiltin()),
            ("LambdaLift",               rx.transform.LambdaLift()),
            ("VMShapeLower",             rx.transform.VMShapeLower()),
            ("AttachGlobalSymbol",       rx.transform.AttachGlobalSymbol()),
            ("AttachExternModules",      rx.transform.AttachExternModules(ext_mods)),
        ]
        for _name, _p in _passes:
            _t0 = _time.perf_counter()
            if verbose:
                print(f"  pipeline pass: {_name} ...", flush=True)
            mod = _p(mod)
            if verbose:
                print(f"  pipeline pass: {_name} done in {_time.perf_counter()-_t0:.1f}s", flush=True)
        return mod

    return pipeline


def build_prefix_block_attention_bias(
    prefix_len: int,
    block_len: int,
    num_attention_heads: int,
    *,
    block_size: int | None = None,
    dtype: str = "float32",
    blocked_value: float = -1e9,
) -> np.ndarray:
    """Build the Fast-dLLM block-causal attention bias (bidirectional within a block)."""
    if prefix_len < 0:
        raise ValueError("prefix_len must be non-negative")
    if block_len <= 0:
        raise ValueError("block_len must be positive")

    total_len = prefix_len + block_len
    effective_block_size = block_len if block_size is None else int(block_size)
    if effective_block_size <= 0:
        raise ValueError("block_size must be positive when provided")

    positions = np.arange(total_len, dtype=np.int32)
    query_blocks = positions[:, None] // effective_block_size
    key_value_blocks = positions[None, :] // effective_block_size
    allowed = query_blocks >= key_value_blocks

    bias_2d = np.where(allowed, 0.0, blocked_value).astype(dtype, copy=False)
    bias = np.broadcast_to(
        bias_2d[None, None, :, :],
        (1, num_attention_heads, total_len, total_len),
    ).copy()

    return bias


def build_cached_attention_bias(
    prefix_len: int,
    block_len: int,
    num_attention_heads: int,
    dtype: str = "float32",
) -> np.ndarray:
    """All-zeros bias for the cache-aware active-block forward.
    Shape: [1, num_heads, block_len, prefix_len + block_len].
    Valid only when prefix_len % block_size == 0 (the active block then sees the whole prefix).
    """
    return np.zeros((1, num_attention_heads, block_len, prefix_len + block_len), dtype=dtype)


def extract_shifted_block_logits(
    logits: np.ndarray,
    *,
    prefix_len: int,
    block_len: int,
) -> np.ndarray:
    """Apply the Fast-dLLM shifted-logit alignment (position i is predicted from logits at i-1)."""
    if logits.ndim != 3:
        raise ValueError(f"Expected logits to have rank 3, got shape {logits.shape}")

    if block_len <= 0:
        raise ValueError("block_len must be positive")

    total_len = logits.shape[1]
    if prefix_len + block_len > total_len:
        raise ValueError(
            f"Requested prefix_len={prefix_len} and block_len={block_len} for logits of length {total_len}"
        )

    if prefix_len > 0:
        source_positions = [prefix_len - 1] + list(range(prefix_len, prefix_len + block_len - 1))
    else:
        source_positions = [0] + list(range(0, block_len - 1))

    return logits[:, source_positions, :]


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = op.split(x, 2, -1)
    return op.concat([op.multiply(x2, Tensor.from_scalar(-1.0, x.dtype)), x1], -1)


def _repeat_kv(hidden_states: Tensor, repeats: int) -> Tensor:
    if repeats == 1:
        return hidden_states
    return op.repeat(hidden_states, repeats, axis=2)


def _cos(x: Tensor, name: str = "cos") -> Tensor:
    return wrap_nested(rx.op.cos(x._expr), name)


def _sin(x: Tensor, name: str = "sin") -> Tensor:
    return wrap_nested(rx.op.sin(x._expr), name)


def _make_rope_tables(seq_len, head_dim: int, rope_theta: float, dtype: str) -> tuple[Tensor, Tensor]:
    half_dim = head_dim // 2
    positions = rx.op.astype(rx.op.arange(0, seq_len, 1, dtype="float32"), "float32")
    exponent = rx.const(-math.log(rope_theta), "float32") * rx.op.arange(
        0, half_dim, 1, dtype="float32"
    )
    exponent = exponent / rx.const(half_dim, "float32")
    inv_freq = rx.op.exp(exponent)
    freqs = rx.op.expand_dims(positions, 1) * rx.op.expand_dims(inv_freq, 0)
    emb = rx.op.concat([freqs, freqs], axis=-1)
    cos = wrap_nested(rx.op.astype(rx.op.expand_dims(rx.op.expand_dims(rx.op.cos(emb), 0), 2), dtype), "cos")
    sin = wrap_nested(rx.op.astype(rx.op.expand_dims(rx.op.expand_dims(rx.op.sin(emb), 0), 2), dtype), "sin")
    return cos, sin


def _apply_rotary_pos_emb(q: Tensor, k: Tensor, rope_theta: float, head_dim: int, dtype: str) -> tuple[Tensor, Tensor]:
    seq_len = q.shape[1]
    cos, sin = _make_rope_tables(seq_len, head_dim, rope_theta, dtype)
    q_embed = op.add(op.multiply(q, cos), op.multiply(_rotate_half(q), sin))
    k_embed = op.add(op.multiply(k, cos), op.multiply(_rotate_half(k), sin))
    return q_embed, k_embed


def _mean(x: Tensor, axis, keepdims: bool, name: str = "mean") -> Tensor:
    return wrap_nested(rx.op.mean(x._expr, axis=axis, keepdims=keepdims), name)


def _rsqrt(x: Tensor, name: str = "rsqrt") -> Tensor:
    return wrap_nested(rx.op.rsqrt(x._expr), name)


class FastDLLMRMSNorm(nn.Module):
    """RMSNorm matching HF's exact precision order: normalize in fp32, cast down to
    the working dtype, THEN multiply by weight (weight multiply is a bf16 x bf16 op).

    Differs from nn.RMSNorm / topi::nn::rms_norm, which multiplies by weight in fp32
    before the final cast — an empirically measured divergence of MAE ~0.0011/call from HF.
    """

    def __init__(self, hidden_size: int, epsilon: float, dtype: str = "float32"):
        self.weight = nn.Parameter((hidden_size,), dtype=dtype)
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        xf = op.astype(x, "float32")
        variance = _mean(op.square(xf), axis=-1, keepdims=True)
        xf = op.multiply(xf, _rsqrt(op.add(variance, Tensor.from_scalar(self.epsilon, "float32"))))
        x_cast = op.astype(xf, input_dtype)
        return op.multiply(self.weight, x_cast)


def _make_rope_tables_with_offset(
    seq_len,
    cache_offset: Tensor,
    head_dim: int,
    rope_theta: float,
    dtype: str,
) -> tuple[Tensor, Tensor]:
    """RoPE tables starting at absolute position cache_offset instead of 0.
    cache_offset is a 0-D int32 Tensor (runtime value).
    """
    half_dim = head_dim // 2
    exponent = rx.const(-math.log(rope_theta), "float32") * rx.op.arange(0, half_dim, 1, dtype="float32")
    exponent = exponent / rx.const(half_dim, "float32")
    inv_freq = rx.op.exp(exponent)
    # positions = arange(0, seq_len) + cache_offset  (broadcast: [seq_len] + [] -> [seq_len])
    pos_rel = rx.op.arange(0, seq_len, 1, dtype="float32")
    cache_off_f32 = rx.op.astype(cache_offset._expr, "float32")
    positions = rx.op.add(pos_rel, cache_off_f32)
    freqs = rx.op.expand_dims(positions, 1) * rx.op.expand_dims(inv_freq, 0)
    emb = rx.op.concat([freqs, freqs], axis=-1)
    cos = wrap_nested(rx.op.astype(rx.op.expand_dims(rx.op.expand_dims(rx.op.cos(emb), 0), 2), dtype), "cos")
    sin = wrap_nested(rx.op.astype(rx.op.expand_dims(rx.op.expand_dims(rx.op.sin(emb), 0), 2), dtype), "sin")
    return cos, sin


def _apply_rotary_pos_emb_with_offset(
    q: Tensor,
    k: Tensor,
    cache_offset: Tensor,
    rope_theta: float,
    head_dim: int,
    dtype: str,
) -> tuple[Tensor, Tensor]:
    """Apply RoPE using absolute positions [cache_offset, cache_offset + seq_len)."""
    seq_len = q.shape[1]
    cos, sin = _make_rope_tables_with_offset(seq_len, cache_offset, head_dim, rope_theta, dtype)
    q_embed = op.add(op.multiply(q, cos), op.multiply(_rotate_half(q), sin))
    k_embed = op.add(op.multiply(k, cos), op.multiply(_rotate_half(k), sin))
    return q_embed, k_embed


def _splice_window(block: Tensor, active: Tensor, replace_position: int, sbs: int, block_size: int) -> Tensor:
    """Functional block-cache update: rebuild the block buffer (axis 2) with `active` at [replace_position, +sbs)."""
    parts: list[Tensor] = []
    if replace_position > 0:
        parts.append(
            wrap_nested(
                rx.op.strided_slice(block._expr, begin=[0], end=[replace_position], strides=[1], axes=[2]),
                "block_before",
            )
        )
    parts.append(active)
    if replace_position + sbs < block_size:
        parts.append(
            wrap_nested(
                rx.op.strided_slice(
                    block._expr, begin=[replace_position + sbs], end=[block_size], strides=[1], axes=[2]
                ),
                "block_after",
            )
        )
    if len(parts) == 1:
        return parts[0]
    return op.concat(parts, dim=2)


def _silu_fp32(x: Tensor, name: str = "silu_fp32") -> Tensor:
    """SiLU computed via fp32 intermediate precision, matching PyTorch's CUDA SiLU
    kernel (opmath convention for reduced-precision dtypes). TVM's op.silu computes
    sigmoid/exp directly in bf16 via the CUDA `hexp` intrinsic — an empirically
    measured divergence of MAE ~0.0021/call from HF.
    """
    input_dtype = x.dtype
    xf = op.astype(x, "float32")
    out = op.multiply(xf, op.sigmoid(xf))
    return op.astype(out, input_dtype)


class FastDLLMRelaxMLP(nn.Module):
    def __init__(self, cfg: FastDLLMRelaxConfig, dtype: str = "float32", fused_gate_up: bool = False):
        self._fused_gate_up = fused_gate_up
        if fused_gate_up:
            self.gate_up_proj = nn.Linear(
                cfg.hidden_size, 2 * cfg.intermediate_size, bias=False, dtype=dtype
            )
        else:
            self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False, dtype=dtype)
            self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        if self._fused_gate_up:
            gate_up = self.gate_up_proj(x)
            gate, up = op.split(gate_up, 2, axis=-1)
            return self.down_proj(_silu_fp32(gate) * up)
        return self.down_proj(_silu_fp32(self.gate_proj(x)) * self.up_proj(x))


class FastDLLMRelaxAttention(nn.Module):
    def __init__(
        self,
        cfg: FastDLLMRelaxConfig,
        dtype: str = "float32",
        fused_qkv: bool = False,
    ):
        self.head_dim = cfg.head_dim
        self.num_q_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.num_kv_groups = cfg.num_attention_heads // cfg.num_key_value_heads
        self.scale = self.head_dim**-0.5
        self.rope_theta = cfg.rope_theta
        self.dtype = dtype
        self.block_size = cfg.bd_size
        self._fused_qkv = fused_qkv

        if fused_qkv:
            self.qkv_proj = nn.Linear(
                cfg.hidden_size,
                (self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim,
                bias=True,
                dtype=dtype,
            )
        else:
            self.q_proj = nn.Linear(
                cfg.hidden_size, self.num_q_heads * self.head_dim, bias=True, dtype=dtype
            )
            self.k_proj = nn.Linear(
                cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=True, dtype=dtype
            )
            self.v_proj = nn.Linear(
                cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=True, dtype=dtype
            )
        self.o_proj = nn.Linear(
            self.num_q_heads * self.head_dim, cfg.hidden_size, bias=False, dtype=dtype
        )

    def _project_qkv(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self._fused_qkv:
            qkv = self.qkv_proj(hidden_states)
            q_dim = self.num_q_heads * self.head_dim
            k_dim = self.num_kv_heads * self.head_dim
            q, k, v = op.split(qkv, [q_dim, q_dim + k_dim], axis=-1)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        return q, k, v

    def forward(self, hidden_states: Tensor, attention_bias: Tensor) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q, k, v = self._project_qkv(hidden_states)

        q = op.reshape(q, (batch_size, seq_len, self.num_q_heads, self.head_dim))
        k = op.reshape(k, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        v = op.reshape(v, (batch_size, seq_len, self.num_kv_heads, self.head_dim))

        q, k = _apply_rotary_pos_emb(q, k, self.rope_theta, self.head_dim, self.dtype)

        k = _repeat_kv(k, self.num_kv_groups)
        v = _repeat_kv(v, self.num_kv_groups)

        q = op.permute_dims(q, axes=[0, 2, 1, 3])
        k = op.permute_dims(k, axes=[0, 2, 3, 1])
        v = op.permute_dims(v, axes=[0, 2, 1, 3])

        scores = op.matmul(q, k, out_dtype="float32")
        scores = op.multiply(scores, Tensor.from_scalar(self.scale, "float32"))
        scores = op.add(scores, attention_bias)
        probs = op.softmax(scores, axis=-1)
        # fp32 through both matmuls, cast once at the very end (matches PyTorch's
        # memory-efficient/math SDPA backends' internal accumulation) --
        # instead of the premature bf16 cast on probs before the V matmul.
        attn_out = op.matmul(probs, op.astype(v, "float32"), out_dtype="float32")
        attn_out = op.astype(attn_out, self.dtype)
        attn_out = op.permute_dims(attn_out, axes=[0, 2, 1, 3])
        attn_out = op.reshape(attn_out, (batch_size, seq_len, self.num_q_heads * self.head_dim))
        return self.o_proj(attn_out)

    def forward_and_collect_kv(self, hidden_states: Tensor, attention_bias: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Same computation as forward() but also returns post-RoPE K/V for prefix caching.
        Returns (attn_output, k_cache, v_cache) where k_cache/v_cache are
        [1, num_kv_heads, seq_len, head_dim].
        """
        batch_size, seq_len, _ = hidden_states.shape

        q, k, v = self._project_qkv(hidden_states)

        q = op.reshape(q, (batch_size, seq_len, self.num_q_heads, self.head_dim))
        k = op.reshape(k, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        v = op.reshape(v, (batch_size, seq_len, self.num_kv_heads, self.head_dim))

        # Prefix always starts at position 0, so standard RoPE applies.
        q, k = _apply_rotary_pos_emb(q, k, self.rope_theta, self.head_dim, self.dtype)

        # Save post-RoPE K/V in cache layout: [1, num_kv_heads, seq_len, head_dim]
        k_cache = op.permute_dims(k, axes=[0, 2, 1, 3])
        v_cache = op.permute_dims(v, axes=[0, 2, 1, 3])

        k_rep = _repeat_kv(k, self.num_kv_groups)
        v_rep = _repeat_kv(v, self.num_kv_groups)

        q_t = op.permute_dims(q, axes=[0, 2, 1, 3])
        k_t = op.permute_dims(k_rep, axes=[0, 2, 3, 1])
        v_t = op.permute_dims(v_rep, axes=[0, 2, 1, 3])

        scores = op.matmul(q_t, k_t, out_dtype="float32")
        scores = op.multiply(scores, Tensor.from_scalar(self.scale, "float32"))
        scores = op.add(scores, attention_bias)
        probs = op.softmax(scores, axis=-1)
        # fp32 through both matmuls, cast once at the very end -- same fix as forward()
        # above. This prefix-building path's own error cascades into every cached
        # generation step, making it the dominant driver of divergence from HF.
        attn_out = op.matmul(probs, op.astype(v_t, "float32"), out_dtype="float32")
        attn_out = op.astype(attn_out, self.dtype)
        attn_out = op.permute_dims(attn_out, axes=[0, 2, 1, 3])
        attn_out = op.reshape(attn_out, (batch_size, seq_len, self.num_q_heads * self.head_dim))
        return self.o_proj(attn_out), k_cache, v_cache

    def forward_cached(
        self,
        hidden_states: Tensor,
        prefix_k: Tensor,
        prefix_v: Tensor,
        cache_offset: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Cache-aware attention forward for a single active block.

        Queries attend to the full key sequence (prefix + active block).
        No attention bias is added — valid because the zero-bias invariant holds
        for any completed-block prefix (prefix_len % block_size == 0).

        Args:
            hidden_states: [1, block_len, hidden_size]
            prefix_k:      [1, num_kv_heads, prefix_len, head_dim]  post-RoPE
            prefix_v:      [1, num_kv_heads, prefix_len, head_dim]  post-RoPE
            cache_offset:  [] int32 scalar — absolute position of first active token (= prefix_len)

        Returns:
            attn_output: [1, block_len, hidden_size]
            full_k:      [1, num_kv_heads, prefix_len + block_len, head_dim]  (cache to store)
            full_v:      [1, num_kv_heads, prefix_len + block_len, head_dim]
        """
        batch_size, block_len, _ = hidden_states.shape

        q, k, v = self._project_qkv(hidden_states)

        q = op.reshape(q, (batch_size, block_len, self.num_q_heads, self.head_dim))
        k = op.reshape(k, (batch_size, block_len, self.num_kv_heads, self.head_dim))
        v = op.reshape(v, (batch_size, block_len, self.num_kv_heads, self.head_dim))

        # RoPE at absolute positions [cache_offset, cache_offset + block_len)
        q, k = _apply_rotary_pos_emb_with_offset(q, k, cache_offset, self.rope_theta, self.head_dim, self.dtype)

        # Active K/V in cache layout (BHSD): [1, num_kv_heads, block_len, head_dim]
        active_k = op.permute_dims(k, axes=[0, 2, 1, 3])
        active_v = op.permute_dims(v, axes=[0, 2, 1, 3])

        # Full K/V cache (BHSD): prefix || active along seq dim (axis=2)
        full_k = op.concat([prefix_k, active_k], dim=2)  # [1, NKV, prefix_len+block_len, HD]
        full_v = op.concat([prefix_v, active_v], dim=2)

        # Convert BHSD cache → BSNH for flash_attn; GQA handled natively
        k_bsnh = op.permute_dims(full_k, axes=[0, 2, 1, 3])
        v_bsnh = op.permute_dims(full_v, axes=[0, 2, 1, 3])
        attn_shape = rx.ShapeExpr([batch_size, block_len, self.num_q_heads, self.head_dim])
        attn_out = wrap_nested(
            rx.call_pure_packed(
                "fast_dllm.flash_attn_bf16",
                q._expr, k_bsnh._expr, v_bsnh._expr,
                sinfo_args=[rx.TensorStructInfo(shape=attn_shape, dtype=self.dtype)],
            ),
            "attn_out",
        )
        attn_out = op.reshape(attn_out, (batch_size, block_len, self.num_q_heads * self.head_dim))
        return self.o_proj(attn_out), full_k, full_v

    def forward_block_partial_perlayer(
        self,
        hidden_states: Tensor,
        prefix_k: Tensor,
        prefix_v: Tensor,
        block_k: Tensor,
        block_v: Tensor,
        cache_offset: Tensor,
        replace_position: int,
        sbs: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Per-layer K/V update partial attention. Matches HF modeling.py:213-219 semantics.

        Splices fresh K/V into block_k at replace_position before this layer's attention,
        so the corrected K propagates through hidden_states to subsequent layers.

        Args:
            hidden_states:    [1, sbs, hidden_size]
            prefix_k:         [1, num_kv_heads, prefix_len, head_dim]
            prefix_v:         [1, num_kv_heads, prefix_len, head_dim]
            block_k:          [1, num_kv_heads, block_size, head_dim]  (stale at replace_position)
            block_v:          [1, num_kv_heads, block_size, head_dim]
            cache_offset:     [] int32  (= prefix_len + replace_position, absolute RoPE start)
            replace_position: int  compile-time constant — start index within block
            sbs:              int  compile-time constant — small_block_size

        Returns:
            attn_output:      [1, sbs, hidden_size]
            active_k:         [1, num_kv_heads, sbs, head_dim]  fresh K for this layer
            active_v:         [1, num_kv_heads, sbs, head_dim]
            updated_block_k:  [1, num_kv_heads, block_size, head_dim]  block_k with fresh K spliced in
            updated_block_v:  [1, num_kv_heads, block_size, head_dim]
        """
        batch_size, _sbs, _ = hidden_states.shape

        q, k, v = self._project_qkv(hidden_states)

        q = op.reshape(q, (batch_size, _sbs, self.num_q_heads, self.head_dim))
        k = op.reshape(k, (batch_size, _sbs, self.num_kv_heads, self.head_dim))
        v = op.reshape(v, (batch_size, _sbs, self.num_kv_heads, self.head_dim))

        q, k = _apply_rotary_pos_emb_with_offset(q, k, cache_offset, self.rope_theta, self.head_dim, self.dtype)

        active_k = op.permute_dims(k, axes=[0, 2, 1, 3])  # [1, kv_heads, sbs, head_dim]
        active_v = op.permute_dims(v, axes=[0, 2, 1, 3])

        updated_block_k = _splice_window(block_k, active_k, replace_position, sbs, self.block_size)
        updated_block_v = _splice_window(block_v, active_v, replace_position, sbs, self.block_size)

        full_k = op.concat([prefix_k, updated_block_k], dim=2)
        full_v = op.concat([prefix_v, updated_block_v], dim=2)
        # Convert BHSD cache → BSNH for flash_attn; GQA handled natively
        k_bsnh = op.permute_dims(full_k, axes=[0, 2, 1, 3])
        v_bsnh = op.permute_dims(full_v, axes=[0, 2, 1, 3])
        attn_shape = rx.ShapeExpr([batch_size, _sbs, self.num_q_heads, self.head_dim])
        attn_out = wrap_nested(
            rx.call_pure_packed(
                "fast_dllm.flash_attn_bf16",
                q._expr, k_bsnh._expr, v_bsnh._expr,
                sinfo_args=[rx.TensorStructInfo(shape=attn_shape, dtype=self.dtype)],
            ),
            "attn_out",
        )
        attn_out = op.reshape(attn_out, (batch_size, _sbs, self.num_q_heads * self.head_dim))
        return self.o_proj(attn_out), active_k, active_v, updated_block_k, updated_block_v


class FastDLLMRelaxDecoderLayer(nn.Module):
    def __init__(
        self,
        cfg: FastDLLMRelaxConfig,
        dtype: str = "float32",
        fused_qkv: bool = False,
        fused_gate_up: bool = False,
    ):
        self.self_attn = FastDLLMRelaxAttention(cfg, dtype=dtype, fused_qkv=fused_qkv)
        self.mlp = FastDLLMRelaxMLP(cfg, dtype=dtype, fused_gate_up=fused_gate_up)
        self.input_layernorm = FastDLLMRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = FastDLLMRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype)

    def forward(self, hidden_states: Tensor, attention_bias: Tensor) -> Tensor:
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states), attention_bias)
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states

    def forward_and_collect_kv(
        self, hidden_states: Tensor, attention_bias: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run layer and return (hidden, k_cache, v_cache) for prefix caching."""
        attn_out, k_cache, v_cache = self.self_attn.forward_and_collect_kv(
            self.input_layernorm(hidden_states), attention_bias
        )
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, k_cache, v_cache

    def forward_cached(
        self,
        hidden_states: Tensor,
        prefix_k: Tensor,
        prefix_v: Tensor,
        cache_offset: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run layer against prefix cache. Returns (hidden, full_k, full_v)."""
        attn_out, full_k, full_v = self.self_attn.forward_cached(
            self.input_layernorm(hidden_states), prefix_k, prefix_v, cache_offset
        )
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, full_k, full_v

    def forward_block_partial_perlayer(
        self,
        hidden_states: Tensor,
        prefix_k: Tensor,
        prefix_v: Tensor,
        block_k: Tensor,
        block_v: Tensor,
        cache_offset: Tensor,
        replace_position: int,
        sbs: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Per-layer K/V update partial forward. Returns (hidden, ak, av, updated_bk, updated_bv)."""
        attn_out, ak, av, ubk, ubv = self.self_attn.forward_block_partial_perlayer(
            self.input_layernorm(hidden_states),
            prefix_k, prefix_v, block_k, block_v, cache_offset, replace_position, sbs,
        )
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, ak, av, ubk, ubv


class FastDLLMRelaxModel(nn.Module):
    def __init__(
        self,
        cfg: FastDLLMRelaxConfig,
        dtype: str = "float32",
        fused_qkv: bool = False,
        fused_gate_up: bool = False,
    ):
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                FastDLLMRelaxDecoderLayer(
                    cfg,
                    dtype=dtype,
                    fused_qkv=fused_qkv,
                    fused_gate_up=fused_gate_up,
                )
                for _ in range(cfg.num_hidden_layers)
            ]
        )
        self.norm = FastDLLMRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype)

    def forward_hidden(self, input_embed: Tensor, attention_bias: Tensor) -> Tensor:
        hidden_states = input_embed
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_bias)
        return self.norm(hidden_states)

    def build_prefix_cache(
        self, prefix_embed: Tensor, prefix_bias: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Run the prefix through all layers and collect per-layer post-RoPE K/V.

        Args:
            prefix_embed: [1, prefix_len, hidden_size]
            prefix_bias:  [1, num_q_heads, prefix_len, prefix_len]  standard block-causal bias

        Returns:
            cached_k: [num_layers, 1, num_kv_heads, prefix_len, head_dim]
            cached_v: [num_layers, 1, num_kv_heads, prefix_len, head_dim]
        """
        hidden_states = prefix_embed
        ks: list[Tensor] = []
        vs: list[Tensor] = []
        for layer in self.layers:
            hidden_states, k_i, v_i = layer.forward_and_collect_kv(hidden_states, prefix_bias)
            ks.append(wrap_nested(rx.op.expand_dims(k_i._expr, axis=0), "k_layer"))
            vs.append(wrap_nested(rx.op.expand_dims(v_i._expr, axis=0), "v_layer"))
        cached_k = op.concat(ks, dim=0)
        cached_v = op.concat(vs, dim=0)
        return cached_k, cached_v

    def forward_cached(
        self,
        active_embed: Tensor,
        cached_k: Tensor,
        cached_v: Tensor,
        cache_offset: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Score the active block against the prefix KV cache.

        Args:
            active_embed: [1, block_len, hidden_size]
            cached_k:     [num_layers, 1, num_kv_heads, prefix_len, head_dim]
            cached_v:     [num_layers, 1, num_kv_heads, prefix_len, head_dim]
            cache_offset: [] int32 scalar  (= prefix_len)

        Returns:
            hidden:       [1, block_len, hidden_size]  (after final norm)
            new_cached_k: [num_layers, 1, num_kv_heads, prefix_len + block_len, head_dim]
            new_cached_v: same shape
        """
        hidden_states = active_embed
        new_ks: list[Tensor] = []
        new_vs: list[Tensor] = []
        for i, layer in enumerate(self.layers):
            # Slice per-layer prefix: [num_layers, 1, kv_heads, prefix_len, head_dim]
            #   -> [1, kv_heads, prefix_len, head_dim]  (take removes the first dim)
            prefix_k_i = wrap_nested(
                rx.op.squeeze(
                    rx.op.strided_slice(cached_k._expr, begin=[i], end=[i + 1], strides=[1], axes=[0]),
                    axis=[0],
                ),
                f"pk_{i}",
            )
            prefix_v_i = wrap_nested(
                rx.op.squeeze(
                    rx.op.strided_slice(cached_v._expr, begin=[i], end=[i + 1], strides=[1], axes=[0]),
                    axis=[0],
                ),
                f"pv_{i}",
            )
            hidden_states, full_k_i, full_v_i = layer.forward_cached(
                hidden_states, prefix_k_i, prefix_v_i, cache_offset
            )
            # full_k_i: [1, kv_heads, prefix_len + block_len, head_dim]
            new_ks.append(wrap_nested(rx.op.expand_dims(full_k_i._expr, axis=0), f"nk_{i}"))
            new_vs.append(wrap_nested(rx.op.expand_dims(full_v_i._expr, axis=0), f"nv_{i}"))
        hidden_states = self.norm(hidden_states)
        new_cached_k = op.concat(new_ks, dim=0)
        new_cached_v = op.concat(new_vs, dim=0)
        return hidden_states, new_cached_k, new_cached_v

    def forward_block_partial_perlayer(
        self,
        active_embed: Tensor,
        prefix_k: Tensor,
        prefix_v: Tensor,
        block_k: Tensor,
        block_v: Tensor,
        cache_offset: Tensor,
        replace_position: int,
        sbs: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Per-layer K/V update partial forward. Achieves exact HF parity.

        Each layer splices fresh K/V into its own block_k slice at replace_position before
        attending, so corrected K propagates through hidden_states to subsequent layers.

        Args:
            active_embed:     [1, sbs, hidden_size]
            prefix_k:         [num_layers, 1, kv_heads, prefix_len, head_dim]
            prefix_v:         [num_layers, 1, kv_heads, prefix_len, head_dim]
            block_k:          [num_layers, 1, kv_heads, block_size, head_dim]  (stale at rp)
            block_v:          [num_layers, 1, kv_heads, block_size, head_dim]
            cache_offset:     [] int32  (= prefix_len + replace_position)
            replace_position: int  compile-time constant
            sbs:              int  compile-time constant

        Returns:
            hidden:       [1, sbs, hidden_size]  (after final norm)
            new_active_k: [num_layers, 1, kv_heads, sbs, head_dim]
            new_block_k:  [num_layers, 1, kv_heads, block_size, head_dim]  (updated per layer)
            new_block_v:  [num_layers, 1, kv_heads, block_size, head_dim]
        """
        hidden_states = active_embed
        new_aks: list[Tensor] = []
        new_bks: list[Tensor] = []
        new_bvs: list[Tensor] = []

        for i, layer in enumerate(self.layers):
            pk_i = wrap_nested(
                rx.op.squeeze(
                    rx.op.strided_slice(prefix_k._expr, begin=[i], end=[i + 1], strides=[1], axes=[0]),
                    axis=[0],
                ),
                f"pk_{i}",
            )
            pv_i = wrap_nested(
                rx.op.squeeze(
                    rx.op.strided_slice(prefix_v._expr, begin=[i], end=[i + 1], strides=[1], axes=[0]),
                    axis=[0],
                ),
                f"pv_{i}",
            )
            bk_i = wrap_nested(
                rx.op.squeeze(
                    rx.op.strided_slice(block_k._expr, begin=[i], end=[i + 1], strides=[1], axes=[0]),
                    axis=[0],
                ),
                f"bk_{i}",
            )
            bv_i = wrap_nested(
                rx.op.squeeze(
                    rx.op.strided_slice(block_v._expr, begin=[i], end=[i + 1], strides=[1], axes=[0]),
                    axis=[0],
                ),
                f"bv_{i}",
            )
            hidden_states, ak_i, _av_i, ubk_i, ubv_i = layer.forward_block_partial_perlayer(
                hidden_states, pk_i, pv_i, bk_i, bv_i, cache_offset, replace_position, sbs,
            )
            new_aks.append(wrap_nested(rx.op.expand_dims(ak_i._expr, axis=0), f"nak_{i}"))
            new_bks.append(wrap_nested(rx.op.expand_dims(ubk_i._expr, axis=0), f"nbk_{i}"))
            new_bvs.append(wrap_nested(rx.op.expand_dims(ubv_i._expr, axis=0), f"nbv_{i}"))

        hidden_states = self.norm(hidden_states)
        new_active_k = op.concat(new_aks, dim=0)
        new_block_k = op.concat(new_bks, dim=0)
        new_block_v = op.concat(new_bvs, dim=0)
        return hidden_states, new_active_k, new_block_k, new_block_v


class FastDLLMRelaxForCausalLM(nn.Module):
    def __init__(
        self,
        cfg: FastDLLMRelaxConfig,
        dtype: str = "float32",
        fused_qkv: bool = False,
        fused_gate_up: bool = False,
    ):
        self.model = FastDLLMRelaxModel(
            cfg, dtype=dtype, fused_qkv=fused_qkv, fused_gate_up=fused_gate_up
        )
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, dtype=dtype)
        self.hidden_size = cfg.hidden_size
        self.num_attention_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.num_hidden_layers = cfg.num_hidden_layers
        self.bd_size = cfg.bd_size
        self.dtype = dtype

    def to(self, dtype: str | None = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def embed(self, input_ids: Tensor) -> Tensor:
        return self.model.embed_tokens(input_ids)

    def forward_hidden(self, input_embed: Tensor, attention_bias: Tensor) -> Tensor:
        return self.model.forward_hidden(input_embed, attention_bias)

    def get_logits(self, hidden_states: Tensor) -> Tensor:
        logits = self.lm_head(hidden_states)
        return logits.astype("float32") if logits.dtype != "float32" else logits

    def build_prefix_cache(
        self, prefix_embed: Tensor, prefix_bias: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Encode completed prefix blocks and return stacked per-layer KV cache.

        Returns:
            cached_k: [num_layers, 1, num_kv_heads, prefix_len, head_dim]
            cached_v: same shape
        """
        return self.model.build_prefix_cache(prefix_embed, prefix_bias)

    def forward_cached(
        self,
        active_embed: Tensor,
        cached_k: Tensor,
        cached_v: Tensor,
        cache_offset: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Score the active block against the prefix KV cache.

        Returns:
            hidden:       [1, block_len, hidden_size]
            new_cached_k: [num_layers, 1, num_kv_heads, prefix_len + block_len, head_dim]
            new_cached_v: same shape
        """
        return self.model.forward_cached(active_embed, cached_k, cached_v, cache_offset)

    def _forward_block_partial_perlayer(
        self,
        active_embed: Tensor,
        prefix_k: Tensor,
        prefix_v: Tensor,
        block_k: Tensor,
        block_v: Tensor,
        cache_offset: Tensor,
        replace_position: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        sbs = self.bd_size // 4
        return self.model.forward_block_partial_perlayer(
            active_embed, prefix_k, prefix_v, block_k, block_v, cache_offset, replace_position, sbs,
        )

    def forward_block_partial_rp0(
        self, active_embed: Tensor, prefix_k: Tensor, prefix_v: Tensor,
        block_k: Tensor, block_v: Tensor, cache_offset: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Per-layer partial forward with replace_position=0."""
        return self._forward_block_partial_perlayer(
            active_embed, prefix_k, prefix_v, block_k, block_v, cache_offset, replace_position=0,
        )

    def forward_block_partial_rp8(
        self, active_embed: Tensor, prefix_k: Tensor, prefix_v: Tensor,
        block_k: Tensor, block_v: Tensor, cache_offset: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Per-layer partial forward with replace_position=8."""
        return self._forward_block_partial_perlayer(
            active_embed, prefix_k, prefix_v, block_k, block_v, cache_offset, replace_position=8,
        )

    def forward_block_partial_rp16(
        self, active_embed: Tensor, prefix_k: Tensor, prefix_v: Tensor,
        block_k: Tensor, block_v: Tensor, cache_offset: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Per-layer partial forward with replace_position=16."""
        return self._forward_block_partial_perlayer(
            active_embed, prefix_k, prefix_v, block_k, block_v, cache_offset, replace_position=16,
        )

    def forward_block_partial_rp24(
        self, active_embed: Tensor, prefix_k: Tensor, prefix_v: Tensor,
        block_k: Tensor, block_v: Tensor, cache_offset: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Per-layer partial forward with replace_position=24."""
        return self._forward_block_partial_perlayer(
            active_embed, prefix_k, prefix_v, block_k, block_v, cache_offset, replace_position=24,
        )

    def get_default_spec(self):
        return nn.spec.ModuleSpec.from_raw(
            {
                "embed": {
                    "input_ids": nn.spec.Tensor(["seq_len"], "int32"),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                },
                "forward_hidden": {
                    "input_embed": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                    "attention_bias": nn.spec.Tensor(
                        [1, self.num_attention_heads, "seq_len", "seq_len"], "float32"
                    ),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                },
                "get_logits": {
                    "hidden_states": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                },
                "build_prefix_cache": {
                    "prefix_embed": nn.spec.Tensor([1, "prefix_len", self.hidden_size], self.dtype),
                    "prefix_bias": nn.spec.Tensor(
                        [1, self.num_attention_heads, "prefix_len", "prefix_len"], "float32"
                    ),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                },
                "forward_cached": {
                    "active_embed": nn.spec.Tensor([1, "block_len", self.hidden_size], self.dtype),
                    "cached_k": nn.spec.Tensor(
                        [self.num_hidden_layers, 1, self.num_kv_heads, "prefix_len", self.head_dim],
                        self.dtype,
                    ),
                    "cached_v": nn.spec.Tensor(
                        [self.num_hidden_layers, 1, self.num_kv_heads, "prefix_len", self.head_dim],
                        self.dtype,
                    ),
                    "cache_offset": nn.spec.Tensor([], "int32"),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                },
                "forward_block_partial_rp0": {
                    "active_embed": nn.spec.Tensor([1, self.bd_size // 4, self.hidden_size], self.dtype),
                    "prefix_k": nn.spec.Tensor(
                        [self.num_hidden_layers, 1, self.num_kv_heads, "prefix_len", self.head_dim],
                        self.dtype,
                    ),
                    "prefix_v": nn.spec.Tensor(
                        [self.num_hidden_layers, 1, self.num_kv_heads, "prefix_len", self.head_dim],
                        self.dtype,
                    ),
                    "block_k": nn.spec.Tensor(
                        [self.num_hidden_layers, 1, self.num_kv_heads, self.bd_size, self.head_dim],
                        self.dtype,
                    ),
                    "block_v": nn.spec.Tensor(
                        [self.num_hidden_layers, 1, self.num_kv_heads, self.bd_size, self.head_dim],
                        self.dtype,
                    ),
                    "cache_offset": nn.spec.Tensor([], "int32"),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                },
                "forward_block_partial_rp8": {
                    "active_embed": nn.spec.Tensor([1, self.bd_size // 4, self.hidden_size], self.dtype),
                    "prefix_k": nn.spec.Tensor(
                        [self.num_hidden_layers, 1, self.num_kv_heads, "prefix_len", self.head_dim],
                        self.dtype,
                    ),
                    "prefix_v": nn.spec.Tensor(
                        [self.num_hidden_layers, 1, self.num_kv_heads, "prefix_len", self.head_dim],
                        self.dtype,
                    ),
                    "block_k": nn.spec.Tensor(
                        [self.num_hidden_layers, 1, self.num_kv_heads, self.bd_size, self.head_dim],
                        self.dtype,
                    ),
                    "block_v": nn.spec.Tensor(
                        [self.num_hidden_layers, 1, self.num_kv_heads, self.bd_size, self.head_dim],
                        self.dtype,
                    ),
                    "cache_offset": nn.spec.Tensor([], "int32"),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                },
                "forward_block_partial_rp16": {
                    "active_embed": nn.spec.Tensor([1, self.bd_size // 4, self.hidden_size], self.dtype),
                    "prefix_k": nn.spec.Tensor(
                        [self.num_hidden_layers, 1, self.num_kv_heads, "prefix_len", self.head_dim],
                        self.dtype,
                    ),
                    "prefix_v": nn.spec.Tensor(
                        [self.num_hidden_layers, 1, self.num_kv_heads, "prefix_len", self.head_dim],
                        self.dtype,
                    ),
                    "block_k": nn.spec.Tensor(
                        [self.num_hidden_layers, 1, self.num_kv_heads, self.bd_size, self.head_dim],
                        self.dtype,
                    ),
                    "block_v": nn.spec.Tensor(
                        [self.num_hidden_layers, 1, self.num_kv_heads, self.bd_size, self.head_dim],
                        self.dtype,
                    ),
                    "cache_offset": nn.spec.Tensor([], "int32"),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                },
                "forward_block_partial_rp24": {
                    "active_embed": nn.spec.Tensor([1, self.bd_size // 4, self.hidden_size], self.dtype),
                    "prefix_k": nn.spec.Tensor(
                        [self.num_hidden_layers, 1, self.num_kv_heads, "prefix_len", self.head_dim],
                        self.dtype,
                    ),
                    "prefix_v": nn.spec.Tensor(
                        [self.num_hidden_layers, 1, self.num_kv_heads, "prefix_len", self.head_dim],
                        self.dtype,
                    ),
                    "block_k": nn.spec.Tensor(
                        [self.num_hidden_layers, 1, self.num_kv_heads, self.bd_size, self.head_dim],
                        self.dtype,
                    ),
                    "block_v": nn.spec.Tensor(
                        [self.num_hidden_layers, 1, self.num_kv_heads, self.bd_size, self.head_dim],
                        self.dtype,
                    ),
                    "cache_offset": nn.spec.Tensor([], "int32"),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                },
            },
            self,
        )
