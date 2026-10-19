"""Relax model definitions: `cuda` (production CUDA/Jetson) and `metal` (in-graph attention)."""

from .cuda import (
    FastDLLMRelaxConfig,
    FastDLLMRelaxForCausalLM,
    build_prefix_block_attention_bias,
    extract_shifted_block_logits,
    opt_fast_dllm_llm_cublas,
)

__all__ = [
    "FastDLLMRelaxConfig",
    "FastDLLMRelaxForCausalLM",
    "build_prefix_block_attention_bias",
    "extract_shifted_block_logits",
    "opt_fast_dllm_llm_cublas",
]
