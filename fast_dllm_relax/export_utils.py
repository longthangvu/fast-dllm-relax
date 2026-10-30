"""Shared helpers for the artifact export scripts (scripts/export_cuda.py, scripts/export_metal.py)."""

import collections
import json

import numpy as np
import torch
import tvm

from .hf_generation import load_fast_dllm
from .model.cuda import FastDLLMRelaxConfig

REQUIRED_FNS = (
    "embed", "forward_hidden", "get_logits", "build_prefix_cache", "forward_cached",
    "forward_block_partial_rp0", "forward_block_partial_rp8",
    "forward_block_partial_rp16", "forward_block_partial_rp24",
)

# KV-cache contract of the exported functions. The runner refuses artifacts whose manifest does not
# name this contract (older artifacts took and returned a functional, stacked BHSD cache).
KV_CACHE_CONTRACT = "inplace_bsnh_v1"
KV_CACHE_MANIFEST = "fast_dllm_relax_nn_kv_cache.json"


def write_kv_cache_manifest(artifact_dir, relax_config: "FastDLLMRelaxConfig", dtype: str) -> None:
    """Record the KV-cache contract and the cache dims the runtime allocates with."""
    manifest = {
        "kv_cache": KV_CACHE_CONTRACT,
        "num_layers": relax_config.num_hidden_layers,
        "num_kv_heads": relax_config.num_key_value_heads,
        "head_dim": relax_config.head_dim,
        "dtype": dtype,
    }
    (artifact_dir / KV_CACHE_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def make_relax_config(hf_config) -> FastDLLMRelaxConfig:
    return FastDLLMRelaxConfig(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        head_dim=getattr(hf_config, "head_dim",
                         hf_config.hidden_size // hf_config.num_attention_heads),
        rms_norm_eps=hf_config.rms_norm_eps,
        rope_theta=float(hf_config.rope_theta),
        context_window_size=hf_config.max_position_embeddings,
        tie_word_embeddings=bool(hf_config.tie_word_embeddings),
        bd_size=int(getattr(hf_config, "bd_size", 32)),
        pad_token_id=int(getattr(hf_config, "pad_token_id", 0)),
    )


def _to_numpy(tensor, torch_dtype, np_dtype):
    """Convert a tensor to a numpy array for npz storage.

    For bfloat16: numpy/npz cannot round-trip ml_dtypes.bfloat16 dtype, so we
    store the raw bits as uint16 instead. A dtype.txt sidecar records the
    semantic dtype so the runner can create correct TVM tensors at load time.
    """
    if torch_dtype == torch.bfloat16:
        t = tensor.detach().to(torch.bfloat16).contiguous()
        return t.view(torch.int16).cpu().numpy().view(np.uint16)
    return tensor.detach().to(torch_dtype).cpu().numpy().astype(np_dtype)


def make_tvm_param(arr, artifact_dtype, dev):
    """Create a TVM tensor from a numpy param array (bf16 params are stored as uint16 bits)."""
    if artifact_dtype == "bfloat16":
        return tvm.runtime.empty(arr.shape, "bfloat16", dev).copyfrom(arr)
    return tvm.runtime.tensor(arr, device=dev)


def remap_params(state_dict, named_param_dict, relax_config, *, torch_dtype, np_dtype):
    """Map HF state_dict weights to fused Relax model param names."""
    mapped = {}
    for name in named_param_dict:
        if name in state_dict:
            mapped[name] = _to_numpy(state_dict[name], torch_dtype, np_dtype)

    # Fuse QKV: q_proj + k_proj + v_proj → qkv_proj
    for layer_idx in range(relax_config.num_hidden_layers):
        attn = f"model.layers.{layer_idx}.self_attn"
        mapped[f"{attn}.qkv_proj.weight"] = np.concatenate([
            _to_numpy(state_dict[f"{attn}.q_proj.weight"], torch_dtype, np_dtype),
            _to_numpy(state_dict[f"{attn}.k_proj.weight"], torch_dtype, np_dtype),
            _to_numpy(state_dict[f"{attn}.v_proj.weight"], torch_dtype, np_dtype),
        ], axis=0)
        mapped[f"{attn}.qkv_proj.bias"] = np.concatenate([
            _to_numpy(state_dict[f"{attn}.q_proj.bias"], torch_dtype, np_dtype),
            _to_numpy(state_dict[f"{attn}.k_proj.bias"], torch_dtype, np_dtype),
            _to_numpy(state_dict[f"{attn}.v_proj.bias"], torch_dtype, np_dtype),
        ], axis=0)

    # Fuse gate+up: gate_proj + up_proj → gate_up_proj
    for layer_idx in range(relax_config.num_hidden_layers):
        mlp = f"model.layers.{layer_idx}.mlp"
        mapped[f"{mlp}.gate_up_proj.weight"] = np.concatenate([
            _to_numpy(state_dict[f"{mlp}.gate_proj.weight"], torch_dtype, np_dtype),
            _to_numpy(state_dict[f"{mlp}.up_proj.weight"], torch_dtype, np_dtype),
        ], axis=0)
        mapped[f"{mlp}.down_proj.weight"] = _to_numpy(state_dict[f"{mlp}.down_proj.weight"], torch_dtype, np_dtype)

    # A tied Relax model has no lm_head.weight param (get_logits reuses embed_tokens.weight), so the
    # loop above already skipped it. An untied model whose checkpoint omits lm_head falls back to
    # the embedding.
    if "lm_head.weight" in named_param_dict and "lm_head.weight" not in mapped:
        mapped["lm_head.weight"] = mapped["model.embed_tokens.weight"]

    missing = set(named_param_dict) - set(mapped)
    if missing:
        raise ValueError(f"Missing params: {sorted(missing)}")
    return mapped


def load_hf_weights(model_name: str, revision: str | None):
    """Load the HF config and a CPU copy of the checkpoint's state_dict."""
    from transformers import AutoConfig  # noqa: PLC0415

    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, revision=revision)
    _, torch_model = load_fast_dllm(model_name, device="auto", trust_remote_code=True, revision=revision)
    torch_model.eval()
    state_dict = {k: v.detach().cpu() for k, v in torch_model.state_dict().items()}
    del torch_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return hf_config, state_dict


def module_kind_counts(mod) -> dict[str, int]:
    """Count runtime module kinds in a compiled module's import tree."""
    counts: collections.Counter = collections.Counter()

    def walk(m):
        counts[m.kind] += 1
        for imported in m.imports:
            walk(imported)

    walk(mod)
    return dict(counts)


def check_byoc_offload(ex) -> dict[str, int]:
    """Fail if BYOC partitioning silently produced no cuBLAS kernels; report CUTLASS usage.

    Wrong pattern ordering or a TVM built without USE_CUBLAS still compiles fine, but runs every
    projection GEMM through generic dlight kernels — slower and numerically further from HF.
    Checked on the in-memory module tree before export. CUTLASS (`static_library` modules) is
    only reported: it claims the batched attention matmuls for float16, but not for bfloat16,
    whose cached-path attention runs in the PyTorch callback anyway.
    """
    kinds = module_kind_counts(ex.mod)
    print(f"  module kinds: {kinds}  (CUTLASS modules: {kinds.get('static_library', 0)})")
    if kinds.get("cublas_json", 0) == 0:
        raise RuntimeError(
            "No cublas_json modules in the compiled artifact: cuBLAS partitioning claimed nothing. "
            "Is TVM built with USE_CUBLAS=ON?"
        )
    return kinds
