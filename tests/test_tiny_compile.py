"""Tiny-config compile checks for both Relax models (random weights, no checkpoint download).

    pytest tests/          # with the TVM env from scripts/setup_tvm.sh

CUDA tests are skipped without a CUDA device; the Metal cross-compile test is skipped when this
TVM build has no Metal codegen.
"""

import numpy as np
import pytest
import tvm
from tvm import relax

from fast_dllm_relax.export_utils import REQUIRED_FNS, check_byoc_offload
from fast_dllm_relax.model.cuda import FastDLLMRelaxConfig, FastDLLMRelaxForCausalLM
from fast_dllm_relax.model.metal import FastDLLMRelaxForCausalLMNative

_has_cuda = tvm.cuda(0).exist
_has_metal_codegen = tvm.get_global_func("target.build.metal", allow_missing=True) is not None
needs_cuda = pytest.mark.skipif(not _has_cuda, reason="no CUDA device")


def tiny_config() -> FastDLLMRelaxConfig:
    # head_dim=128: the attention callback hard-codes softmax scale 1/sqrt(128).
    return FastDLLMRelaxConfig(
        vocab_size=64, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=128, rms_norm_eps=1e-6,
        rope_theta=1e6, context_window_size=256, tie_word_embeddings=False,
        bd_size=32, pad_token_id=0,
    )


def _export(cls, dtype):
    model = cls(tiny_config(), dtype=dtype, fused_qkv=True, fused_gate_up=True)
    return model.export_tvm(spec=model.get_default_spec())


def test_exported_functions_and_param_lists_match():
    mod_c, named_c = _export(FastDLLMRelaxForCausalLM, "float16")
    mod_n, named_n = _export(FastDLLMRelaxForCausalLMNative, "float16")
    for mod in (mod_c, mod_n):
        names = {gv.name_hint for gv in mod.get_global_vars()}
        assert set(REQUIRED_FNS) <= names
    # Same names/order/shapes, so one params.npz layout serves both models.
    sig = lambda named: [(n, [int(d) for d in p.shape], str(p.dtype)) for n, p in named]  # noqa: E731
    assert sig(named_c) == sig(named_n)


def test_native_model_has_no_attention_callback():
    mod_n, _ = _export(FastDLLMRelaxForCausalLMNative, "float16")
    assert "flash_attn" not in mod_n.script()
    mod_c, _ = _export(FastDLLMRelaxForCausalLM, "float16")
    assert "fast_dllm.flash_attn_bf16" in mod_c.script()


@pytest.mark.skipif(not _has_metal_codegen, reason="TVM built without Metal codegen")
def test_metal_cross_compile():
    from fast_dllm_relax.targets.metal import metal_target

    mod_n, _ = _export(FastDLLMRelaxForCausalLMNative, "float16")
    tvm.compile(mod_n, metal_target(), relax_pipeline=relax.get_pipeline("opt_fast_dllm_llm_native"))


@needs_cuda
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_cuda_pipeline_offloads_to_cublas(dtype):
    import fast_dllm_relax.runtime  # noqa: F401  registers the attention callback

    mod_c, _ = _export(FastDLLMRelaxForCausalLM, dtype)
    target = tvm.target.Target.from_device(tvm.cuda(0))
    ex = tvm.compile(
        mod_c, target, relax_pipeline=relax.get_pipeline("opt_fast_dllm_llm_cublas", verbose=False)
    )
    kinds = check_byoc_offload(ex)
    if dtype == "float16":
        # CUTLASS claims the batched fp16 attention matmuls (not the bf16 ones).
        assert kinds.get("static_library", 0) > 0


@needs_cuda
def test_native_matches_callback_attention_on_cuda():
    """In-graph attention (Metal model) vs the torch-callback attention (CUDA model), same pipeline."""
    import fast_dllm_relax.runtime  # noqa: F401  registers the attention callback

    cfg = tiny_config()
    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)
    pipeline = relax.get_pipeline("opt_fast_dllm_llm_native")
    mod_n, named = _export(FastDLLMRelaxForCausalLMNative, "float16")
    mod_c, _ = _export(FastDLLMRelaxForCausalLM, "float16")
    vm_n = relax.VirtualMachine(tvm.compile(mod_n, target, relax_pipeline=pipeline), dev)
    vm_c = relax.VirtualMachine(tvm.compile(mod_c, target, relax_pipeline=pipeline), dev)

    rng = np.random.default_rng(0)
    params_np = []
    for name, p in named:
        shape = [int(d) for d in p.shape]
        if name.endswith("norm.weight"):
            params_np.append(np.ones(shape, dtype="float16"))  # keep activations in a sane range
        else:
            params_np.append((0.02 * rng.standard_normal(shape)).astype("float16"))
    params = [tvm.runtime.tensor(a, device=dev) for a in params_np]

    NL, NKV, HD, HS = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, cfg.hidden_size
    PLEN, BSIZ, SBS = 64, cfg.bd_size, cfg.bd_size // 4
    t = lambda shape: tvm.runtime.tensor(rng.standard_normal(shape).astype("float16"), dev)  # noqa: E731
    active, small = t((1, BSIZ, HS)), t((1, SBS, HS))
    ck, cv = t((NL, 1, NKV, PLEN, HD)), t((NL, 1, NKV, PLEN, HD))
    bk, bv = t((NL, 1, NKV, BSIZ, HD)), t((NL, 1, NKV, BSIZ, HD))
    off = lambda v: tvm.runtime.tensor(np.asarray(v, dtype="int32"), dev)  # noqa: E731

    def run(vm):
        outs = [vm["forward_cached"](active, ck, cv, off(PLEN), params)]
        for rp in (0, 8, 16, 24):
            outs.append(vm[f"forward_block_partial_rp{rp}"](small, ck, cv, bk, bv, off(PLEN + rp), params))
        return [[r.numpy().astype("float32") for r in o] for o in outs]

    for outs_n, outs_c in zip(run(vm_n), run(vm_c)):
        for a, b in zip(outs_n, outs_c):
            assert np.isfinite(a).all() and np.isfinite(b).all()
            np.testing.assert_allclose(a, b, rtol=1e-2, atol=1e-2)
