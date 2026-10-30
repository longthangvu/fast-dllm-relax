"""Native-attention Relax model + pipeline for non-CUDA targets (Metal).

Sibling of the CUDA model (`model/cuda.py`) that changes no CUDA behavior. The CUDA model computes
`forward_cached` and `forward_block_partial_perlayer` attention through the
`fast_dllm.flash_attn_bf16` Python callback (torch SDPA/flash-attn, `runtime.py`), which cannot run
on Metal. This module overrides only those two attention methods so attention is computed in-graph:
  concat(prefix, active/updated block) -> repeat GQA -> fp32 QK^T -> softmax -> fp32 PV -> cast.
No bias is needed: the active block sees the whole block-aligned prefix
(`model/cuda.py` `build_cached_attention_bias` is all zeros), and the callback likewise attends with
causal=False and no bias.

The module attribute tree is identical to the CUDA model (`model.layers[i].self_attn.*`), so the
exported parameter names/order -- and therefore `remap_params` and the npz/param_names artifact
format -- are unchanged. Exported functions are the same ten, plus four static-shape variants
(`METAL_STATIC_FNS`) whose fixed row counts compile to faster Metal kernels.

The module also defines the device-kernel int32 index narrowing (`metal_tir_pipeline`) used for
Metal builds, with its total-length cap `NARROW_I32_MAX_TOTAL_LEN`.

Build with `scripts/export_metal.py` (on a Linux workstation or on the Mac).

NOT PORTED to the in-place KV cache (plan 003). The CUDA model's cache contract changed: one
preallocated BSNH cache written in place through the `fast_dllm.attn_inplace` / `fast_dllm.kv_write`
callbacks. This module still overrides the old functional-cache methods, so it no longer exports;
`scripts/export_metal.py` refuses to run and the runner refuses Metal artifacts until a port (an
in-graph row write via call_tir_inplace plus attention over a sliced view) lands.
"""

import tvm
from tvm import relax as rx
from tvm import s_tir, tirx
from tvm.relax import register_pipeline
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
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
    _splice_window,
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
    "forward_cached_b32", "forward_cached_block_b32", "get_logits_m8", "get_logits_m32",
    "get_confidence_m8", "get_confidence_m32",
)


def _stack_kernel(shapes: list, dtype: str) -> tirx.PrimFunc:
    """Opaque (never fused) concat of [1, ...] tensors along axis 0; see
    FastDLLMRelaxModelNative._stack_layers. Symbolic dims (prefix_len) become shape vars bound
    from the buffers."""
    from tvm import te, topi  # pylint: disable=import-outside-toplevel

    sym: dict[str, tirx.Var] = {}

    def dim(d):
        if isinstance(d, int):
            return d
        if isinstance(d, tirx.IntImm):
            return int(d)
        return sym.setdefault(str(d), tirx.Var(str(d), "int64"))

    inputs = [te.placeholder([dim(d) for d in shape], dtype, name=f"layer{i}") for i, shape in enumerate(shapes)]
    out = topi.concatenate(inputs, axis=0)
    return te.create_prim_func([*inputs, out]).with_attr("op_pattern", 8)  # kOpaque


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
        prefix_k: Tensor,
        prefix_v: Tensor,
        cache_offset: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Same contract as FastDLLMRelaxAttention.forward_cached, in-graph attention."""
        batch_size, block_len, _ = hidden_states.shape

        q, k, v = self._project_qkv(hidden_states)
        q = op.reshape(q, (batch_size, block_len, self.num_q_heads, self.head_dim))
        k = op.reshape(k, (batch_size, block_len, self.num_kv_heads, self.head_dim))
        v = op.reshape(v, (batch_size, block_len, self.num_kv_heads, self.head_dim))

        q, k = _apply_rotary_pos_emb_with_offset(q, k, cache_offset, self.rope_theta, self.head_dim, self.dtype)

        active_k = op.permute_dims(k, axes=[0, 2, 1, 3])
        active_v = op.permute_dims(v, axes=[0, 2, 1, 3])

        full_k = op.concat([prefix_k, active_k], dim=2)
        full_v = op.concat([prefix_v, active_v], dim=2)

        attn_out = self._attend_native(q, full_k, full_v)
        return self.o_proj(attn_out), full_k, full_v, active_k, active_v

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
        """Same contract as FastDLLMRelaxAttention.forward_block_partial_perlayer, in-graph attention."""
        batch_size, _sbs, _ = hidden_states.shape

        q, k, v = self._project_qkv(hidden_states)
        q = op.reshape(q, (batch_size, _sbs, self.num_q_heads, self.head_dim))
        k = op.reshape(k, (batch_size, _sbs, self.num_kv_heads, self.head_dim))
        v = op.reshape(v, (batch_size, _sbs, self.num_kv_heads, self.head_dim))

        q, k = _apply_rotary_pos_emb_with_offset(q, k, cache_offset, self.rope_theta, self.head_dim, self.dtype)

        active_k = op.permute_dims(k, axes=[0, 2, 1, 3])
        active_v = op.permute_dims(v, axes=[0, 2, 1, 3])

        updated_block_k = _splice_window(block_k, active_k, replace_position, sbs, self.block_size)
        updated_block_v = _splice_window(block_v, active_v, replace_position, sbs, self.block_size)

        full_k = op.concat([prefix_k, updated_block_k], dim=2)
        full_v = op.concat([prefix_v, updated_block_v], dim=2)

        attn_out = self._attend_native(q, full_k, full_v)
        return self.o_proj(attn_out), active_k, active_v, updated_block_k, updated_block_v


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

    def _stack_layers(self, per_layer: list[Tensor], name: str) -> Tensor:
        """Stack per-layer K/V with an opaque copy kernel instead of a fusible concat.

        FuseOps (Relax) does not enforce the target's max_function_args, so the 28-input concat
        can absorb a layer's RoPE producers and exceed Metal's 31 buffer slots (observed with
        --gemm-acc float32: 31 tensors + the scalar-argument buffer). As an opaque call it keeps
        exactly num_layers inputs + 1 output (+ the scalar buffer); the copy is bit-exact.
        """
        return op.tensor_ir_op(
            _stack_kernel([t.shape for t in per_layer], per_layer[0].dtype),
            name,
            args=per_layer,
            out=Tensor.placeholder([len(per_layer), *per_layer[0].shape[1:]], per_layer[0].dtype),
        )


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
        cached_k: Tensor,
        cached_v: Tensor,
        cache_offset: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """forward_cached with a static bd_size-row active block (block append, chunked prefill)."""
        return self.model.forward_cached(active_embed, cached_k, cached_v, cache_offset)

    def forward_cached_block_b32(
        self,
        active_embed: Tensor,
        cached_k: Tensor,
        cached_v: Tensor,
        cache_offset: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """forward_cached_block with a static bd_size-row active block (refinement, loglikelihood)."""
        return self.model.forward_cached_block(active_embed, cached_k, cached_v, cache_offset)

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
        for name, base in (("forward_cached_b32", "forward_cached"), ("forward_cached_block_b32", "forward_cached_block")):
            spec = dict(raw[base])
            spec["active_embed"] = nn.spec.Tensor([1, self.bd_size, self.hidden_size], self.dtype)
            raw[name] = spec
        for name, rows in (
            ("get_logits_m8", self.bd_size // 4), ("get_logits_m32", self.bd_size),
            ("get_confidence_m8", self.bd_size // 4), ("get_confidence_m32", self.bd_size),
        ):
            spec = dict(raw["get_logits"])
            spec["hidden_states"] = nn.spec.Tensor([1, rows, self.hidden_size], self.dtype)
            raw[name] = spec
        return raw
