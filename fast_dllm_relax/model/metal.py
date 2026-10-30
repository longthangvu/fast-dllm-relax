"""Native-attention Relax model + pipeline for non-CUDA targets (Metal).

Sibling of the CUDA model (`model/cuda.py`) that changes no CUDA behavior. Both models share the
in-place KV-cache contract (`export_utils.KV_CACHE_CONTRACT`): one preallocated cache per K and V,
[num_layers, 1, cap, num_kv_heads, head_dim] (BSNH rows), written in place by every cache function.
The CUDA model writes and attends through Python callbacks (`fast_dllm.attn_inplace` /
`fast_dllm.kv_write`, torch flash-attn/SDPA in `runtime.py`), which cannot run on Metal. This module
does both in the graph:
  - `kv_write_kernel`, called through call_tir_inplace, writes a layer's post-RoPE K/V rows;
  - attention slices rows [0, prefix_len + attend_rel) of that layer out of the written cache into
    contiguous BHSD K/V (the same tensors the functional cache's concat(prefix, active) produced),
    then runs GQA attention in-graph: fp32 QK^T -> softmax -> fp32 PV -> cast. No bias is needed:
    the active block sees the whole block-aligned prefix, as in the CUDA callback.

The module attribute tree is identical to the CUDA model (`model.layers[i].self_attn.*`), so the
exported parameter names/order -- and therefore `remap_params` and the npz/param_names artifact
format -- are unchanged. Exported functions are the CUDA set (`export_utils.REQUIRED_FNS`) plus the
static-shape variants in `METAL_STATIC_FNS`, whose fixed row counts compile to faster Metal kernels.

The module also defines the device-kernel int32 index narrowing (`metal_tir_pipeline`) used for
Metal builds, with its total-length cap `NARROW_I32_MAX_TOTAL_LEN`.

Build with `scripts/export_metal.py` (on a Linux workstation or on the Mac).
"""

import tvm
from tvm import relax as rx
from tvm import s_tir, tirx
from tvm.relax import register_pipeline
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.relax.frontend.nn.core import wrap_nested
from tvm.script import tirx as T
from tvm.s_tir import dlight
from tvm.s_tir.dlight.analysis import get_root_block
from tvm.s_tir.dlight.gpu.matmul import MetalMatmul, get_in_out_dtypes, get_reduction_blocks

from .cuda import (
    FastDLLMRelaxAttention,
    FastDLLMRelaxConfig,
    FastDLLMRelaxDecoderLayer,
    FastDLLMRelaxForCausalLM,
    FastDLLMRelaxModel,
    _apply_rotary_pos_emb_with_offset,
)


def _cfg(bx: int, by: int, bk: int, ty: int, tz: int) -> dict:
    return dict(block_size_x=bx, block_size_y=by, block_size_k=bk, ty_len=ty, tz_len=tz, vector_size=4)


# MetalMatmul tile configs per static GEMM shape (M, N, K) of Fast_dLLM_v2_1.5B (hidden 1536,
# intermediate 8960, fused qkv N=2048, fused gate_up N=17920), picked by a timing sweep on an M4.
# MetalMatmul accumulates
# K in order for every config, so these are bit-identical to the default config; shapes not listed
# (other models, dynamic M) keep the default.
METAL_MATMUL_CONFIGS: dict[tuple[int, int, int], dict] = {
    (8, 1536, 8960): _cfg(8, 8, 64, 1, 1),      # down_proj, partial forwards
    (32, 1536, 8960): _cfg(16, 16, 64, 1, 2),   # down_proj, full block
    (8, 2048, 1536): _cfg(8, 8, 64, 1, 1),      # qkv
    (32, 2048, 1536): _cfg(16, 16, 64, 1, 2),
    (8, 1536, 1536): _cfg(8, 8, 32, 1, 4),      # o_proj
    (32, 1536, 1536): _cfg(16, 8, 32, 1, 1),
    (8, 17920, 1536): _cfg(8, 32, 64, 1, 1),    # gate_up
    (32, 17920, 1536): _cfg(32, 16, 64, 1, 4),
    (8, 151936, 1536): _cfg(8, 8, 64, 2, 4),    # tied lm_head (get_logits / get_confidence)
    (32, 151936, 1536): _cfg(32, 16, 64, 1, 4),
}


def _static_mnk(block: tirx.SBlock) -> tuple[int, int, int] | None:
    """(M, N, K) of a matmul block C[..., M, N] += A[..., M, K] * B[N, K] with static extents:
    N = last spatial iter, M = product of the other spatial iters, K = product of the reduction
    iters. None if any extent is symbolic."""
    spatial, reduce = [], []
    for iv in block.iter_vars:
        if not isinstance(iv.dom.extent, tirx.IntImm):
            return None
        (reduce if iv.iter_type == tirx.IterVar.CommReduce else spatial).append(int(iv.dom.extent))
    if len(spatial) < 2 or not reduce:
        return None
    m = 1
    for e in spatial[:-1]:
        m *= e
    k = 1
    for e in reduce:
        k *= e
    return m, spatial[-1], k


class MatmulMetalMixedPrecision(dlight.gpu.Matmul):
    """dlight.gpu.Matmul, except mixed-precision matmuls (input dtype != output dtype) never reach
    the generic Matmul schedule: they get MetalMatmul (simdgroup MMA, float16 x float16 -> float32
    accumulators) on Metal, else Fallback.

    The attention score matmuls are fp16 x fp16 -> fp32 (`out_dtype="float32"`, in both the
    prefix path `forward_and_collect_kv` and `_attend_native`). The generic Matmul() on those
    batched dynamic-shape kernels hangs (tirx.Simplify never terminates on
    `fused_matmul_multiply*_add*` during lowering). Matmul() itself tries MetalMatmul first on
    Metal, but before the patched intrinsics MetalMatmul only had float16 accumulators, so mixed
    blocks fell through to the generic schedule. The CUDA pipeline avoids the hang because
    CUTLASS consumes those matmuls; the Jetson path drops Matmul() entirely (model/cuda.py
    opt_fast_dllm_llm_cublas, use_cutlass=False). Letting them fall through the rule list instead
    fails: GeneralReduction raises a ScheduleError on them rather than declining.
    The same-dtype matmuls (fp16 linear layers, fp32 attention probs x V) take Matmul() as before,
    which uses MetalMatmul on Metal, except that shapes in METAL_MATMUL_CONFIGS get their tuned
    MetalMatmul tiles.
    """

    def apply(self, func, target, tunable):
        if not isinstance(func, tirx.PrimFunc) or not self.is_target_available(target):
            return None
        sch = s_tir.Schedule(func)
        reduction_blocks = get_reduction_blocks(sch, sch.get_child_blocks(get_root_block(sch)))
        if reduction_blocks is not None:
            in_dtype, out_dtype = get_in_out_dtypes(sch.get(reduction_blocks[0]))
            cfg = METAL_MATMUL_CONFIGS.get(_static_mnk(sch.get(reduction_blocks[0])))
            if in_dtype != out_dtype:
                if target.kind.name == "metal":
                    try:
                        mma = MetalMatmul(**(cfg or {})).apply(func, target, tunable)
                    except Exception:  # noqa: BLE001 -- not tensorizable: use Fallback
                        mma = None
                    if mma is not None:
                        return mma
                return dlight.gpu.Fallback().apply(func, target, tunable)
            if cfg is not None and target.kind.name == "metal":
                try:
                    tuned = MetalMatmul(**cfg).apply(func, target, tunable)
                except Exception:  # noqa: BLE001 -- fall back to the default schedule
                    tuned = None
                if tuned is not None:
                    return tuned
        return super().apply(func, target, tunable)


@tvm.transform.module_pass(opt_level=0, name="AccumulateMatmulFP32")
def accumulate_matmul_fp32(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
    """Rewrite every float16 x float16 -> float16 (or bfloat16 x bfloat16 -> bfloat16)
    relax.matmul (the projection GEMMs and the tied lm_head; the attention matmuls already
    request float32) into a float32-output matmul cast back to the input dtype. On Metal this
    selects simdgroup float32 accumulators (the mixed-precision MetalMatmul path); the stored
    activations keep the model dtype."""
    from tvm.relax.dpl import is_op, rewrite_call, wildcard  # pylint: disable=import-outside-toplevel

    pattern = is_op("relax.matmul")(wildcard(), wildcard())

    def rewriter(expr, _matches):
        a, b = expr.args
        dtype = a.struct_info.dtype
        out = str(expr.attrs.out_dtype)
        if dtype not in ("float16", "bfloat16") or b.struct_info.dtype != dtype or out not in ("", "void", dtype):
            return expr
        return rx.op.astype(rx.op.matmul(a, b, out_dtype="float32"), dtype)

    updates = {
        gv: rewrite_call(pattern, rewriter, func)
        for gv, func in mod.functions.items()
        if isinstance(func, rx.Function)
    }
    for gv, func in updates.items():
        mod[gv] = func
    return mod


@register_pipeline("opt_fast_dllm_llm_native")
def opt_fast_dllm_llm_native(ext_mods: list[nn.ExternModule] | None = None, gemm_acc: str = "float32"):
    """Generic dlight-only pipeline (no cuBLAS/CUTLASS, no CUDA graph) for the Metal build.

    Legalize -> fuse -> dlight schedule -> VM lowering, plus ComputePrimValue before VMShapeLower.
    Uses MatmulMetalMixedPrecision in place of dlight.gpu.Matmul (see its docstring). The same
    pipeline also compiles for CUDA, so the Metal graph's numerics can be checked on a workstation.
    """
    ext_mods = ext_mods or []
    if gemm_acc not in ("float16", "float32"):
        raise ValueError(f"gemm_acc must be float16 or float32, got {gemm_acc!r}")
    acc_passes = [accumulate_matmul_fp32] if gemm_acc == "float32" else []

    @tvm.transform.module_pass(opt_level=0)
    def pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        return tvm.transform.Sequential(
            [
                *acc_passes,
                rx.transform.FuseTransposeMatmul(),
                rx.transform.LegalizeOps(),
                rx.transform.AnnotateTIROpPattern(),
                rx.transform.FoldConstant(),
                rx.transform.FuseOps(),
                rx.transform.FuseTIR(),
                rx.transform.DeadCodeElimination(),
                dlight.ApplyDefaultSchedule(
                    MatmulMetalMixedPrecision(),
                    dlight.gpu.GEMV(),
                    dlight.gpu.Reduction(),
                    dlight.gpu.GeneralReduction(),
                    dlight.gpu.Fallback(),
                ),
                rx.transform.RewriteDataflowReshape(),
                rx.transform.ToNonDataflow(),
                rx.transform.RemovePurityChecking(),
                rx.transform.CallTIRRewrite(),
                rx.transform.StaticPlanBlockMemory(),
                rx.transform.LowerAllocTensor(),
                rx.transform.KillAfterLastUse(),
                rx.transform.LowerRuntimeBuiltin(),
                rx.transform.ComputePrimValue(),
                rx.transform.VMShapeLower(),
                rx.transform.AttachGlobalSymbol(),
                rx.transform.AttachExternModules(ext_mods),
            ]
        )(mod)

    return pipeline


# Total-length cap (prompt + generated + block padding) for artifacts built with narrow_i32: the
# largest dynamic buffer is the full-sequence softmax [1, 12, L, L], whose flat index passes int32
# at L = 13,378. 12,288 = 384 blocks of 32 leaves ~8% headroom.
NARROW_I32_MAX_TOTAL_LEN = 12288


@tvm.transform.module_pass(opt_level=0, name="NarrowDeviceIndexToInt32")
def narrow_device_index_i32(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
    """ForceNarrowIndexToInt32 on device kernels only.

    Relax shape vars are int64 and the default NarrowDataType(32) never narrows function params, so
    every symbolic-shape Metal kernel indexes in (emulated) 64-bit -- 3-16x slower than int32.
    Runs after the s_tir pipeline (post SplitHostDevice/LowerDeviceKernelLaunch): host code keeps
    int64 shape math, and the runtime converts each int64 scalar arg to the kernel's declared int32
    from the device FunctionInfo (pack_args.h INT64_TO_INT32). Does not prove bounds: callers must
    keep total length <= NARROW_I32_MAX_TOTAL_LEN.
    """
    funcs = {}
    for gv, func in mod.functions.items():
        if isinstance(func, tirx.PrimFunc) and int(func.attrs.get("calling_conv", 0)) == 2:  # kDeviceKernelLaunch
            func = tirx.transform.ForceNarrowIndexToInt32()(tvm.IRModule({gv: func}))[gv]
        funcs[gv] = func
    return tvm.IRModule(funcs, attrs=mod.attrs)


def metal_tir_pipeline(narrow_i32: bool):
    """TIR pipeline for tvm.compile: default s_tir, plus device-kernel int32 narrowing if requested."""
    if not narrow_i32:
        return "default"
    return tvm.transform.Sequential([tvm.tirx.get_tir_pipeline("s_tir"), narrow_device_index_i32])


# Static-shape siblings of the dynamic functions, Metal artifacts only. The runtime uses one when
# the row count matches and the artifact has it, and falls back to the dynamic function otherwise.
METAL_STATIC_FNS = (
    "forward_cached_b32", "get_logits_m8", "get_logits_m32", "get_confidence_m8", "get_confidence_m32",
)


def kv_write_kernel(num_layers: int, num_kv_heads: int, head_dim: int, dtype: str) -> tirx.PrimFunc:
    """In-place KV-cache row write: k_cache[layer, 0, start + r] = k[0, r] (same for v), r < rows.

    Called through call_tir_inplace (`_kv_write`), so the caches are both inputs and outputs and
    nothing is copied besides the written rows. `layer` and `start` are scalar args, so every layer
    and row offset shares this one kernel; `rows` and `cap` bind from the buffers. Self-scheduled
    (one threadgroup per (row, kv head), one thread per head_dim element) and opaque, so FuseOps
    keeps it a separate kernel. No bounds check: the runtime checks every write window before the
    call (`runtime._check_write`).
    """
    n_heads, n_dim = T.int64(num_kv_heads), T.int64(head_dim)

    @T.prim_func(private=True)
    def kv_write(
        var_k: T.handle, var_v: T.handle, var_k_cache: T.handle, var_v_cache: T.handle,
        layer: T.int64, start: T.int64,
    ):
        T.func_attr({"op_pattern": 8, "tirx.noalias": True, "tirx.is_scheduled": True})
        rows, cap = T.int64(), T.int64()
        k = T.match_buffer(var_k, (T.int64(1), rows, n_heads, n_dim), dtype)
        v = T.match_buffer(var_v, (T.int64(1), rows, n_heads, n_dim), dtype)
        k_cache = T.match_buffer(var_k_cache, (T.int64(num_layers), T.int64(1), cap, n_heads, n_dim), dtype)
        v_cache = T.match_buffer(var_v_cache, (T.int64(num_layers), T.int64(1), cap, n_heads, n_dim), dtype)
        for bx in T.thread_binding(rows * n_heads, thread="blockIdx.x"):
            for tx in T.thread_binding(n_dim, thread="threadIdx.x"):
                with T.sblock("kv_write"):
                    vr = T.axis.spatial(rows, bx // n_heads)
                    vh = T.axis.spatial(n_heads, bx % n_heads)
                    vd = T.axis.spatial(n_dim, tx)
                    k_cache[layer, 0, start + vr, vh, vd] = k[0, vr, vh, vd]
                    v_cache[layer, 0, start + vr, vh, vd] = v[0, vr, vh, vd]

    return kv_write


def _kv_write(k: Tensor, v: Tensor, k_cache: Tensor, v_cache: Tensor, layer: int, start) -> tuple[Tensor, Tensor]:
    """Write k / v ([1, rows, num_kv_heads, head_dim]) at cache rows [start, start + rows) of `layer`.
    Returns the written caches (aliases of the inputs); readers must use these so the write is
    ordered before them and is not dead code."""
    num_layers, _, _, num_kv_heads, head_dim = k_cache.shape
    as_expr = lambda x: tirx.IntImm("int64", x) if isinstance(x, int) else x  # noqa: E731
    return op.tensor_ir_inplace_op(
        kv_write_kernel(int(num_layers), int(num_kv_heads), int(head_dim), k_cache.dtype),
        "kv_write",
        args=[k, v, k_cache, v_cache, as_expr(layer), as_expr(start)],
        inplace_indices=[2, 3],
        out=[Tensor.placeholder(k_cache.shape, k_cache.dtype), Tensor.placeholder(v_cache.shape, v_cache.dtype)],
    )


def _layer_rows_bhsd(cache: Tensor, layer: int, rows) -> Tensor:
    """Cache rows [0, rows) of `layer` as a contiguous BHSD tensor [1, num_kv_heads, rows, head_dim].
    assume_inbound keeps the slice's extent `rows` (FuseTIR cannot match the clamped form)."""
    _, _, _, num_kv_heads, head_dim = cache.shape
    sliced = wrap_nested(
        rx.op.strided_slice(cache._expr, axes=[0, 2], begin=[layer, 0], end=[layer + 1, rows], assume_inbound=True),
        "cache_rows",
    )
    return op.permute_dims(op.reshape(sliced, (1, rows, num_kv_heads, head_dim)), axes=[0, 2, 1, 3])


def confidence_kernel(rows: int, vocab: int, threads: int = 1024) -> tirx.PrimFunc:
    """Per-row first-index argmax and max softmax probability of fp32 logits [1, rows, vocab].

    Hand-scheduled (dlight has no rule for a standalone max/min reduction: Reduction takes only
    sums, GeneralReduction needs an injective epilogue, so those fall to a one-thread-per-row
    Fallback that is ~100x slower). One threadgroup per row makes three passes over the row --
    max, first index equal to the max, sum(exp(x - max)) -- each a cross-thread reduction lowered
    to SIMD shuffles + threadgroup memory. maxp = 1 / sum, which is softmax(x).max() exactly
    in real arithmetic; it is closer to the float64 value than PyTorch's CPU softmax (~1e-7 vs
    ~5e-5 relative error at vocab 151936).
    """
    steps = (vocab + threads - 1) // threads

    @T.prim_func(private=True)
    def confidence(
        logits: T.Buffer((1, rows, vocab), "float32"),
        ids: T.Buffer((1, rows), "int32"),
        maxp: T.Buffer((1, rows), "float32"),
    ):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        row_max = T.sblock_alloc_buffer((rows,), "float32", scope="shared")
        row_arg = T.sblock_alloc_buffer((rows,), "int32", scope="shared")
        row_sum = T.sblock_alloc_buffer((rows,), "float32", scope="shared")
        for r in T.thread_binding(rows, thread="blockIdx.x"):
            for tx in T.thread_binding(threads, thread="threadIdx.x"):
                for ko in range(steps):
                    with T.sblock("max"):
                        vr = T.axis.spatial(rows, r)
                        vk = T.axis.reduce(vocab, ko * threads + tx)
                        T.where(ko * threads + tx < vocab)
                        with T.init():
                            row_max[vr] = T.min_value("float32")
                        row_max[vr] = T.max(row_max[vr], logits[0, vr, vk])
            for tx in T.thread_binding(threads, thread="threadIdx.x"):
                for ko in range(steps):
                    with T.sblock("argmax"):
                        vr = T.axis.spatial(rows, r)
                        vk = T.axis.reduce(vocab, ko * threads + tx)
                        T.where(ko * threads + tx < vocab)
                        with T.init():
                            row_arg[vr] = T.max_value("int32")
                        row_arg[vr] = T.min(row_arg[vr], T.Select(logits[0, vr, vk] == row_max[vr], vk, vocab))
            for tx in T.thread_binding(threads, thread="threadIdx.x"):
                for ko in range(steps):
                    with T.sblock("expsum"):
                        vr = T.axis.spatial(rows, r)
                        vk = T.axis.reduce(vocab, ko * threads + tx)
                        T.where(ko * threads + tx < vocab)
                        with T.init():
                            row_sum[vr] = T.float32(0)
                        row_sum[vr] = row_sum[vr] + T.exp(logits[0, vr, vk] - row_max[vr])
            for tx in T.thread_binding(threads, thread="threadIdx.x"):
                with T.sblock("out"):
                    vr = T.axis.spatial(rows, r)
                    T.where(tx == 0)
                    ids[0, vr] = row_arg[vr]
                    maxp[0, vr] = T.float32(1) / row_sum[vr]

    return confidence


class FastDLLMRelaxAttentionNative(FastDLLMRelaxAttention):
    def _attend_native(self, q: Tensor, full_k: Tensor, full_v: Tensor) -> Tensor:
        """In-graph attention, no bias.

        Grouped-query attention without materializing repeated K/V: the num_kv_groups query heads
        that share a KV head (query head h uses KV head h // num_kv_groups, as op.repeat would)
        are stacked along the row axis, so each KV head serves its group in one batched matmul.
        Every output element is the same dot product as with repeated K/V.

        Args:
            q:      [1, q_len, num_q_heads, head_dim]  post-RoPE
            full_k: [1, num_kv_heads, kv_len, head_dim]  (BHSD)
            full_v: [1, num_kv_heads, kv_len, head_dim]

        Returns:
            attn_out: [1, q_len, num_q_heads * head_dim]  (before o_proj)
        """
        batch_size, q_len, _, _ = q.shape
        group_rows = self.num_kv_groups * q_len

        q_t = op.permute_dims(q, axes=[0, 2, 1, 3])            # [b, q_heads, q_len, dim]
        q_g = op.reshape(q_t, (batch_size, self.num_kv_heads, group_rows, self.head_dim))
        k_t = op.permute_dims(full_k, axes=[0, 1, 3, 2])       # [b, kv_heads, dim, kv_len]

        scores = op.matmul(q_g, k_t, out_dtype="float32")      # [b, kv_heads, group * q_len, kv_len]
        scores = op.multiply(scores, Tensor.from_scalar(self.scale, "float32"))
        probs = op.softmax(scores, axis=-1)
        attn_out = op.matmul(probs, op.astype(full_v, "float32"), out_dtype="float32")
        attn_out = op.astype(attn_out, self.dtype)
        attn_out = op.reshape(attn_out, (batch_size, self.num_q_heads, q_len, self.head_dim))
        attn_out = op.permute_dims(attn_out, axes=[0, 2, 1, 3])
        return op.reshape(attn_out, (batch_size, q_len, self.num_q_heads * self.head_dim))

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
        """Same contract as FastDLLMRelaxAttention.forward_cached, in-graph: write this call's
        post-RoPE K/V at rows [prefix_len + write_rel, + seq_len) of this layer's cache, then attend
        over rows [0, prefix_len + attend_rel) of the written cache. Returns (attn_output, k_cache,
        v_cache), the caches being the written aliases."""
        batch_size, seq_len, _ = hidden_states.shape

        q, k, v = self._project_qkv(hidden_states)
        q = op.reshape(q, (batch_size, seq_len, self.num_q_heads, self.head_dim))
        k = op.reshape(k, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        v = op.reshape(v, (batch_size, seq_len, self.num_kv_heads, self.head_dim))

        q, k = _apply_rotary_pos_emb_with_offset(q, k, cache_offset, self.rope_theta, self.head_dim, self.dtype)

        k_cache, v_cache = _kv_write(k, v, k_cache, v_cache, layer, prefix_len + write_rel)
        full_k = _layer_rows_bhsd(k_cache, layer, prefix_len + attend_rel)
        full_v = _layer_rows_bhsd(v_cache, layer, prefix_len + attend_rel)

        attn_out = self._attend_native(q, full_k, full_v)
        return self.o_proj(attn_out), k_cache, v_cache


class FastDLLMRelaxDecoderLayerNative(FastDLLMRelaxDecoderLayer):
    def __init__(
        self, cfg: FastDLLMRelaxConfig, dtype: str = "float32", fused_qkv: bool = False, fused_gate_up: bool = False,
    ):
        super().__init__(cfg, dtype=dtype, fused_qkv=fused_qkv, fused_gate_up=fused_gate_up)
        self.self_attn = FastDLLMRelaxAttentionNative(cfg, dtype=dtype, fused_qkv=fused_qkv)


class FastDLLMRelaxModelNative(FastDLLMRelaxModel):
    def __init__(
        self, cfg: FastDLLMRelaxConfig, dtype: str = "float32", fused_qkv: bool = False, fused_gate_up: bool = False,
    ):
        super().__init__(cfg, dtype=dtype, fused_qkv=fused_qkv, fused_gate_up=fused_gate_up)
        self.layers = nn.ModuleList(
            [
                FastDLLMRelaxDecoderLayerNative(cfg, dtype=dtype, fused_qkv=fused_qkv, fused_gate_up=fused_gate_up)
                for _ in range(cfg.num_hidden_layers)
            ]
        )

    def build_prefix_cache(
        self, prefix_embed: Tensor, prefix_bias: Tensor, k_cache: Tensor, v_cache: Tensor
    ) -> tuple[Tensor, Tensor]:
        """FastDLLMRelaxModel.build_prefix_cache with the per-layer K/V written in-graph (`_kv_write`)
        instead of through the fast_dllm.kv_write callback."""
        hidden_states = prefix_embed
        for i, layer in enumerate(self.layers):
            hidden_states, k_i, v_i = layer.forward_and_collect_kv(hidden_states, prefix_bias)
            k_cache, v_cache = _kv_write(k_i, v_i, k_cache, v_cache, i, 0)
        return k_cache, v_cache


class FastDLLMRelaxForCausalLMNative(FastDLLMRelaxForCausalLM):
    def __init__(
        self, cfg: FastDLLMRelaxConfig, dtype: str = "float32", fused_qkv: bool = False, fused_gate_up: bool = False,
    ):
        super().__init__(cfg, dtype=dtype, fused_qkv=fused_qkv, fused_gate_up=fused_gate_up)
        self.model = FastDLLMRelaxModelNative(cfg, dtype=dtype, fused_qkv=fused_qkv, fused_gate_up=fused_gate_up)

    # Static-M variants. Every decode caller passes exactly bd_size (or bd_size // 4) rows; a static
    # M lets the projection and lm_head GEMMs compile to int32-indexed Metal kernels, which run
    # several times faster than the symbolic-M ones. prefix_len stays symbolic.

    def forward_cached_b32(
        self,
        active_embed: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        cache_offset: Tensor,
        prefix_len: int,
    ) -> Tensor:
        """forward_cached with a static bd_size-row active block (refinement, block-append,
        chunked prefill, scoring)."""
        return self.model.forward_cached(active_embed, k_cache, v_cache, cache_offset, prefix_len)

    def get_logits_m8(self, hidden_states: Tensor) -> Tensor:
        """get_logits for the bd_size // 4-row partial path, static M."""
        return self.get_logits(hidden_states)

    def get_logits_m32(self, hidden_states: Tensor) -> Tensor:
        """get_logits for the bd_size-row full-block path, static M."""
        return self.get_logits(hidden_states)

    def get_confidence_m8(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """get_confidence for the bd_size // 4-row partial path, static M."""
        return self._confidence(hidden_states)

    def get_confidence_m32(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """get_confidence for the bd_size-row full-block path, static M."""
        return self._confidence(hidden_states)

    def _confidence(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """Per-row greedy token and its softmax probability, reduced on the device so the runtime
        reads back 2 x rows values instead of the [1, rows, vocab] fp32 logits.

        Returns (unshifted; the runtime applies the shift):
            ids:  [1, rows] int32  argmax over the vocab, first index on ties (as torch.argmax)
            maxp: [1, rows] fp32   max softmax probability = 1 / sum(exp(logits - max))
        """
        logits = self.get_logits(hidden_states)                          # [1, rows, vocab] fp32
        _, rows, vocab = (int(d) for d in logits.shape)
        return op.tensor_ir_op(
            confidence_kernel(rows, vocab),
            "confidence",
            args=[logits],
            out=[Tensor.placeholder((1, rows), "int32"), Tensor.placeholder((1, rows), "float32")],
        )

    def _default_spec_raw(self) -> dict:
        raw = super()._default_spec_raw()
        spec = dict(raw["forward_cached"])
        spec["active_embed"] = nn.spec.Tensor([1, self.bd_size, self.hidden_size], self.dtype)
        raw["forward_cached_b32"] = spec
        for name, rows in (
            ("get_logits_m8", self.bd_size // 4), ("get_logits_m32", self.bd_size),
            ("get_confidence_m8", self.bd_size // 4), ("get_confidence_m32", self.bd_size),
        ):
            spec = dict(raw["get_logits"])
            spec["hidden_states"] = nn.spec.Tensor([1, rows, self.hidden_size], self.dtype)
            raw[name] = spec
        return raw
