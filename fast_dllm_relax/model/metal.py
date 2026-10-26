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
format -- are unchanged. Exported functions are the same ten (`get_default_spec` is inherited).

Build with `scripts/export_metal.py`.
"""

import tvm
from tvm import relax as rx
from tvm import s_tir, tirx
from tvm.relax import register_pipeline
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.s_tir import dlight
from tvm.s_tir.dlight.analysis import get_root_block
from tvm.s_tir.dlight.gpu.matmul import get_in_out_dtypes, get_reduction_blocks

from .cuda import (
    FastDLLMRelaxAttention,
    FastDLLMRelaxConfig,
    FastDLLMRelaxDecoderLayer,
    FastDLLMRelaxForCausalLM,
    FastDLLMRelaxModel,
    _apply_rotary_pos_emb_with_offset,
    _splice_window,
)


class MatmulSkipMixedPrecision(dlight.gpu.Matmul):
    """dlight.gpu.Matmul, except mixed-precision matmuls (input dtype != output dtype) get Fallback.

    The attention score matmuls are fp16 x fp16 -> fp32 (`out_dtype="float32"`, in both the
    prefix path `forward_and_collect_kv` and `_attend_native`). Matmul() on those
    batched dynamic-shape kernels hangs (tirx.Simplify never terminates on
    `fused_matmul_multiply*_add*` during lowering).
    The CUDA pipeline avoids it because CUTLASS consumes those matmuls; the Jetson path drops Matmul()
    entirely (model/cuda.py opt_fast_dllm_llm_cublas, use_cutlass=False) but still has cuBLAS for the
    linear layers. Here there is no cuBLAS, so keep Matmul() for the same-dtype fp16 linear-layer
    GEMMs and schedule only the mixed-precision ones with dlight.gpu.Fallback directly (letting
    them fall through the rule list instead fails: GeneralReduction raises a ScheduleError on them
    rather than declining).
    """

    def apply(self, func, target, tunable):
        if not isinstance(func, tirx.PrimFunc) or not self.is_target_available(target):
            return None
        sch = s_tir.Schedule(func)
        reduction_blocks = get_reduction_blocks(sch, sch.get_child_blocks(get_root_block(sch)))
        if reduction_blocks is not None:
            in_dtype, out_dtype = get_in_out_dtypes(sch.get(reduction_blocks[0]))
            if in_dtype != out_dtype:
                return dlight.gpu.Fallback().apply(func, target, tunable)
        return super().apply(func, target, tunable)


@register_pipeline("opt_fast_dllm_llm_native")
def opt_fast_dllm_llm_native(ext_mods: list[nn.ExternModule] | None = None):
    """Generic dlight-only pipeline (no cuBLAS/CUTLASS, no CUDA graph) for the Metal build.

    Legalize -> fuse -> dlight schedule -> VM lowering, plus ComputePrimValue before VMShapeLower.
    Uses MatmulSkipMixedPrecision in place of dlight.gpu.Matmul (see its docstring). The same
    pipeline also compiles for CUDA, so the Metal graph's numerics can be checked on a workstation.
    """
    ext_mods = ext_mods or []

    @tvm.transform.module_pass(opt_level=0)
    def pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        return tvm.transform.Sequential(
            [
                rx.transform.FuseTransposeMatmul(),
                rx.transform.LegalizeOps(),
                rx.transform.AnnotateTIROpPattern(),
                rx.transform.FoldConstant(),
                rx.transform.FuseOps(),
                rx.transform.FuseTIR(),
                rx.transform.DeadCodeElimination(),
                dlight.ApplyDefaultSchedule(
                    MatmulSkipMixedPrecision(),
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


class FastDLLMRelaxAttentionNative(FastDLLMRelaxAttention):
    def _attend_native(self, q: Tensor, full_k: Tensor, full_v: Tensor) -> Tensor:
        """In-graph attention, no bias.

        Args:
            q:      [1, q_len, num_q_heads, head_dim]  post-RoPE
            full_k: [1, num_kv_heads, kv_len, head_dim]  (BHSD)
            full_v: [1, num_kv_heads, kv_len, head_dim]

        Returns:
            attn_out: [1, q_len, num_q_heads * head_dim]  (before o_proj)
        """
        batch_size, q_len, _, _ = q.shape
        if self.num_kv_groups == 1:
            full_k_rep, full_v_rep = full_k, full_v
        else:
            full_k_rep = op.repeat(full_k, self.num_kv_groups, axis=1)
            full_v_rep = op.repeat(full_v, self.num_kv_groups, axis=1)

        q_t = op.permute_dims(q, axes=[0, 2, 1, 3])            # BHSD: [b, heads, q_len, dim]
        k_t = op.permute_dims(full_k_rep, axes=[0, 1, 3, 2])   # BHDS: [b, heads, dim, kv_len]

        scores = op.matmul(q_t, k_t, out_dtype="float32")
        scores = op.multiply(scores, Tensor.from_scalar(self.scale, "float32"))
        probs = op.softmax(scores, axis=-1)
        attn_out = op.matmul(probs, op.astype(full_v_rep, "float32"), out_dtype="float32")
        attn_out = op.astype(attn_out, self.dtype)
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


class FastDLLMRelaxForCausalLMNative(FastDLLMRelaxForCausalLM):
    def __init__(
        self, cfg: FastDLLMRelaxConfig, dtype: str = "float32", fused_qkv: bool = False, fused_gate_up: bool = False,
    ):
        super().__init__(cfg, dtype=dtype, fused_qkv=fused_qkv, fused_gate_up=fused_gate_up)
        self.model = FastDLLMRelaxModelNative(cfg, dtype=dtype, fused_qkv=fused_qkv, fused_gate_up=fused_gate_up)
