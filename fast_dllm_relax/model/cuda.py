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
        # dlight.gpu.Matmul() schedules the batched bf16 -> fp32 prefill attention matmuls, but on
        # the fp16 -> fp32 QK^T matmul the scheduled kernel never finishes compiling (TIR codegen).
        # With CUTLASS on, CUTLASS (fp16 only, no bf16) consumes that one before scheduling; with
        # use_cutlass=False (Jetson) drop Matmul() so the attention matmuls go to
        # dlight.gpu.GeneralReduction() instead (slower per kernel, but terminates).
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
    *,
    block_size: int | None = None,
    dtype: str = "float32",
    blocked_value: float = -1e9,
) -> np.ndarray:
    """Build the Fast-dLLM block-causal attention bias (bidirectional within a block).

    Shape [1, 1, T, T] (T = prefix_len + block_len): every head uses the same mask, and the model
    broadcasts it over heads when adding it to the scores.
    """
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
    return bias_2d[None, None, :, :]


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
        Returns (attn_output, k, v) where k/v are [1, seq_len, num_kv_heads, head_dim] (BSNH, the
        KV cache's row layout).
        """
        batch_size, seq_len, _ = hidden_states.shape

        q, k, v = self._project_qkv(hidden_states)

        q = op.reshape(q, (batch_size, seq_len, self.num_q_heads, self.head_dim))
        k = op.reshape(k, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        v = op.reshape(v, (batch_size, seq_len, self.num_kv_heads, self.head_dim))

        # Prefix always starts at position 0, so standard RoPE applies.
        q, k = _apply_rotary_pos_emb(q, k, self.rope_theta, self.head_dim, self.dtype)

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
        return self.o_proj(attn_out), k, v

    def forward_cached(
        self,
        hidden_states: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        cache_offset: Tensor,
        layer: int,
        prefix_len,
        write_rel: int,
        attend_rel,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Cache-aware attention for the active rows, against the in-place KV cache.

        The `fast_dllm.attn_inplace` callback (runtime.py) writes this call's post-RoPE K/V into
        rows [prefix_len + write_rel, + seq_len) of this layer's cache slice, then attends over rows
        [0, prefix_len + attend_rel) -- prefix and active block, no bias, no copies. No attention
        bias is needed because the prefix is block-aligned (the active block sees all of it).

        Args:
            hidden_states: [1, seq_len, hidden_size]
            k_cache:       [num_layers, 1, cap, num_kv_heads, head_dim]  (BSNH rows, written in place)
            v_cache:       same shape
            cache_offset:  [] int32 -- absolute position of the first active row (RoPE)
            layer:         compile-time layer index
            prefix_len:    symbolic int -- committed cache rows
            write_rel:     compile-time row offset of the active rows within the block
            attend_rel:    rows of the block attended to (seq_len, or bd_size for partial steps)

        Returns:
            attn_output: [1, seq_len, hidden_size]
            k_cache, v_cache: aliases of the inputs (threaded to the next layer, which orders the
            in-place writes)
        """
        batch_size, seq_len, _ = hidden_states.shape

        q, k, v = self._project_qkv(hidden_states)

        q = op.reshape(q, (batch_size, seq_len, self.num_q_heads, self.head_dim))
        k = op.reshape(k, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        v = op.reshape(v, (batch_size, seq_len, self.num_kv_heads, self.head_dim))

        # RoPE at absolute positions [cache_offset, cache_offset + seq_len)
        q, k = _apply_rotary_pos_emb_with_offset(q, k, cache_offset, self.rope_theta, self.head_dim, self.dtype)

        # BSNH in, BSNH out; GQA handled natively by the attention kernel
        attn_shape = rx.ShapeExpr([batch_size, seq_len, self.num_q_heads, self.head_dim])
        k_cache, v_cache, attn_out = wrap_nested(
            rx.op.call_inplace_packed(
                "fast_dllm.attn_inplace",
                q._expr, k._expr, v._expr, k_cache._expr, v_cache._expr,
                rx.PrimValue(layer), rx.PrimValue(prefix_len), rx.PrimValue(write_rel), rx.PrimValue(attend_rel),
                inplace_indices=[3, 4, -1],
                sinfo_args=[
                    k_cache._expr.struct_info,
                    v_cache._expr.struct_info,
                    rx.TensorStructInfo(shape=attn_shape, dtype=self.dtype),
                ],
            ),
            "attn_inplace",
        )
        attn_out = op.reshape(attn_out, (batch_size, seq_len, self.num_q_heads * self.head_dim))
        return self.o_proj(attn_out), k_cache, v_cache


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
        """Run layer and return (hidden, k, v) for prefix caching (k/v BSNH, see the attention's)."""
        attn_out, k, v = self.self_attn.forward_and_collect_kv(
            self.input_layernorm(hidden_states), attention_bias
        )
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, k, v

    def forward_cached(
        self,
        hidden_states: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        cache_offset: Tensor,
        layer: int,
        prefix_len,
        write_rel: int,
        attend_rel,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run layer against the in-place KV cache. Returns (hidden, k_cache, v_cache)."""
        attn_out, k_cache, v_cache = self.self_attn.forward_cached(
            self.input_layernorm(hidden_states), k_cache, v_cache, cache_offset,
            layer, prefix_len, write_rel, attend_rel,
        )
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, k_cache, v_cache


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
        self, prefix_embed: Tensor, prefix_bias: Tensor, k_cache: Tensor, v_cache: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Run the prefix through all layers, writing per-layer post-RoPE K/V into cache rows [0, prefix_len).

        Args:
            prefix_embed: [1, prefix_len, hidden_size]
            prefix_bias:  [1, 1, prefix_len, prefix_len]  block-causal bias (broadcast over heads)
            k_cache:      [num_layers, 1, cap, num_kv_heads, head_dim]  (written in place)
            v_cache:      same shape

        Returns:
            k_cache, v_cache: aliases of the inputs (returned so the writes stay live)
        """
        hidden_states = prefix_embed
        for i, layer in enumerate(self.layers):
            hidden_states, k_i, v_i = layer.forward_and_collect_kv(hidden_states, prefix_bias)
            k_cache, v_cache = wrap_nested(
                rx.op.call_inplace_packed(
                    "fast_dllm.kv_write",
                    k_i._expr, v_i._expr, k_cache._expr, v_cache._expr,
                    rx.PrimValue(i), rx.PrimValue(0),
                    inplace_indices=[2, 3],
                    sinfo_args=[k_cache._expr.struct_info, v_cache._expr.struct_info],
                ),
                f"kv_write_{i}",
            )
        return k_cache, v_cache

    def forward_cached(
        self,
        active_embed: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        cache_offset: Tensor,
        prefix_len,
        write_rel: int = 0,
        attend_rel=None,
    ) -> Tensor:
        """Score active rows against the in-place KV cache (see FastDLLMRelaxAttention.forward_cached).

        Each layer writes the active rows' K/V at cache rows [prefix_len + write_rel, + seq_len) and
        attends over rows [0, prefix_len + attend_rel). attend_rel defaults to seq_len (a full block
        at the cache frontier: block-append, refinement full steps, scoring, chunked prefill); the
        partial per-layer forwards pass write_rel=replace_position, attend_rel=bd_size.

        Returns:
            hidden: [1, seq_len, hidden_size]  (after final norm)
        """
        if attend_rel is None:
            attend_rel = active_embed.shape[1]
        hidden_states = active_embed
        for i, layer in enumerate(self.layers):
            hidden_states, k_cache, v_cache = layer.forward_cached(
                hidden_states, k_cache, v_cache, cache_offset, i, prefix_len, write_rel, attend_rel
            )
        return self.norm(hidden_states)


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
        # Tied checkpoints (Fast-dLLM v2 1.5B) reuse embed_tokens.weight in get_logits instead of
        # carrying a second copy of the vocab x hidden matrix as an lm_head parameter.
        self.tie_word_embeddings = cfg.tie_word_embeddings
        if not self.tie_word_embeddings:
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
        if self.tie_word_embeddings:
            # Same IR nn.Linear(bias=False).forward emits (matmul with permuted weight), so the cuBLAS
            # pattern claims it exactly as it claims an untied lm_head.
            logits = op.matmul(hidden_states, op.permute_dims(self.model.embed_tokens.weight))
        else:
            logits = self.lm_head(hidden_states)
        return logits.astype("float32") if logits.dtype != "float32" else logits

    def build_prefix_cache(
        self, prefix_embed: Tensor, prefix_bias: Tensor, k_cache: Tensor, v_cache: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Encode completed prefix blocks into KV cache rows [0, prefix_len) in place.

        Returns:
            k_cache, v_cache: the (written) inputs, [num_layers, 1, cap, num_kv_heads, head_dim]
        """
        return self.model.build_prefix_cache(prefix_embed, prefix_bias, k_cache, v_cache)

    def forward_cached(
        self,
        active_embed: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        cache_offset: Tensor,
        prefix_len: int,
    ) -> Tensor:
        """Score the active block against the prefix KV cache, writing its K/V at rows
        [prefix_len, prefix_len + block_len) (committed by the caller advancing prefix_len, or
        overwritten by the next call).

        Returns:
            hidden: [1, block_len, hidden_size]
        """
        return self.model.forward_cached(active_embed, k_cache, v_cache, cache_offset, prefix_len)

    def _forward_block_partial_perlayer(
        self,
        active_embed: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        cache_offset: Tensor,
        prefix_len: int,
        replace_position: int,
    ) -> Tensor:
        """Partial forward of one small block at replace_position: each layer overwrites the block's
        rows [replace_position, +sbs) with fresh K/V, then attends over prefix + the whole block (the
        other block rows keep the last full step's K/V) -- the HF per-layer block-cache update.
        """
        return self.model.forward_cached(
            active_embed, k_cache, v_cache, cache_offset, prefix_len,
            write_rel=replace_position, attend_rel=self.bd_size,
        )

    def forward_block_partial_rp0(
        self, active_embed: Tensor, k_cache: Tensor, v_cache: Tensor, cache_offset: Tensor, prefix_len: int,
    ) -> Tensor:
        """Per-layer partial forward with replace_position=0."""
        return self._forward_block_partial_perlayer(
            active_embed, k_cache, v_cache, cache_offset, prefix_len, replace_position=0,
        )

    def forward_block_partial_rp8(
        self, active_embed: Tensor, k_cache: Tensor, v_cache: Tensor, cache_offset: Tensor, prefix_len: int,
    ) -> Tensor:
        """Per-layer partial forward with replace_position=8."""
        return self._forward_block_partial_perlayer(
            active_embed, k_cache, v_cache, cache_offset, prefix_len, replace_position=8,
        )

    def forward_block_partial_rp16(
        self, active_embed: Tensor, k_cache: Tensor, v_cache: Tensor, cache_offset: Tensor, prefix_len: int,
    ) -> Tensor:
        """Per-layer partial forward with replace_position=16."""
        return self._forward_block_partial_perlayer(
            active_embed, k_cache, v_cache, cache_offset, prefix_len, replace_position=16,
        )

    def forward_block_partial_rp24(
        self, active_embed: Tensor, k_cache: Tensor, v_cache: Tensor, cache_offset: Tensor, prefix_len: int,
    ) -> Tensor:
        """Per-layer partial forward with replace_position=24."""
        return self._forward_block_partial_perlayer(
            active_embed, k_cache, v_cache, cache_offset, prefix_len, replace_position=24,
        )

    def get_default_spec(self):
        return nn.spec.ModuleSpec.from_raw(self._default_spec_raw(), self)

    def _kv_cache_spec(self) -> nn.spec.Tensor:
        """In-place KV cache: [num_layers, 1, cap, num_kv_heads, head_dim], BSNH rows."""
        return nn.spec.Tensor(
            [self.num_hidden_layers, 1, "cap", self.num_kv_heads, self.head_dim], self.dtype
        )

    def _default_spec_raw(self) -> dict:
        """Raw spec dict for get_default_spec (split out so subclasses can add entries)."""
        partial = {
            "active_embed": nn.spec.Tensor([1, self.bd_size // 4, self.hidden_size], self.dtype),
            "k_cache": self._kv_cache_spec(),
            "v_cache": self._kv_cache_spec(),
            "cache_offset": nn.spec.Tensor([], "int32"),
            "prefix_len": nn.spec.Int(),
            "$": {"param_mode": "packed", "effect_mode": "none"},
        }
        return {
            "embed": {
                "input_ids": nn.spec.Tensor(["seq_len"], "int32"),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "forward_hidden": {
                "input_embed": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "attention_bias": nn.spec.Tensor([1, 1, "seq_len", "seq_len"], "float32"),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "get_logits": {
                "hidden_states": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "build_prefix_cache": {
                "prefix_embed": nn.spec.Tensor([1, "prefix_len", self.hidden_size], self.dtype),
                "prefix_bias": nn.spec.Tensor([1, 1, "prefix_len", "prefix_len"], "float32"),
                "k_cache": self._kv_cache_spec(),
                "v_cache": self._kv_cache_spec(),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "forward_cached": {
                "active_embed": nn.spec.Tensor([1, "block_len", self.hidden_size], self.dtype),
                "k_cache": self._kv_cache_spec(),
                "v_cache": self._kv_cache_spec(),
                "cache_offset": nn.spec.Tensor([], "int32"),
                "prefix_len": nn.spec.Int(),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "forward_block_partial_rp0": dict(partial),
            "forward_block_partial_rp8": dict(partial),
            "forward_block_partial_rp16": dict(partial),
            "forward_block_partial_rp24": dict(partial),
        }
