"""Tiny-config compile checks for both Relax models (random weights, no checkpoint download).

    pytest tests/          # with the TVM env from scripts/setup_tvm.sh

CUDA tests are skipped without a CUDA device; the Metal cross-compile test is skipped when this
TVM build has no Metal codegen.
"""

import functools

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


def tiny_config(tied: bool = False) -> FastDLLMRelaxConfig:
    # head_dim=128: the attention callback hard-codes softmax scale 1/sqrt(128).
    return FastDLLMRelaxConfig(
        vocab_size=64, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=128, rms_norm_eps=1e-6,
        rope_theta=1e6, context_window_size=256, tie_word_embeddings=tied,
        bd_size=32, pad_token_id=0,
    )


def _export(cls, dtype, tied: bool = False):
    model = cls(tiny_config(tied), dtype=dtype, fused_qkv=True, fused_gate_up=True)
    return model.export_tvm(spec=model.get_default_spec())


def _random_params(named, seed: int = 0) -> dict[str, np.ndarray]:
    """fp16 random weights by name (norm weights = 1 to keep activations in a sane range)."""
    rng = np.random.default_rng(seed)
    params = {}
    for name, p in named:
        shape = [int(d) for d in p.shape]
        if name.endswith("norm.weight"):
            params[name] = np.ones(shape, dtype="float16")
        else:
            params[name] = (0.02 * rng.standard_normal(shape)).astype("float16")
    return params


@functools.cache
def _compiled_cublas(tied: bool):
    """(VirtualMachine, named params) for the tiny fp16 CUDA model, production pipeline."""
    import fast_dllm_relax.runtime  # noqa: F401, PLC0415  registers the attention callback

    dev = tvm.cuda(0)
    mod, named = _export(FastDLLMRelaxForCausalLM, "float16", tied=tied)
    ex = tvm.compile(
        mod,
        tvm.target.Target.from_device(dev),
        relax_pipeline=relax.get_pipeline("opt_fast_dllm_llm_cublas", verbose=False),
    )
    return ex, relax.VirtualMachine(ex, dev), named


@pytest.mark.parametrize("tied", [False, True])
def test_exported_functions_and_param_lists_match(tied):
    mod_c, named_c = _export(FastDLLMRelaxForCausalLM, "float16", tied=tied)
    mod_n, named_n = _export(FastDLLMRelaxForCausalLMNative, "float16", tied=tied)
    for mod in (mod_c, mod_n):
        names = {gv.name_hint for gv in mod.get_global_vars()}
        assert set(REQUIRED_FNS) <= names
    # Same names/order/shapes, so one params.npz layout serves both models.
    sig = lambda named: [(n, [int(d) for d in p.shape], str(p.dtype)) for n, p in named]  # noqa: E731
    assert sig(named_c) == sig(named_n)
    # A tied model reuses embed_tokens.weight in get_logits instead of carrying lm_head.weight.
    assert ("lm_head.weight" in dict(named_c)) == (not tied)
    assert "model.embed_tokens.weight" in dict(named_c)


def test_native_model_has_no_attention_callback():
    mod_n, _ = _export(FastDLLMRelaxForCausalLMNative, "float16")
    assert "flash_attn" not in mod_n.script()
    mod_c, _ = _export(FastDLLMRelaxForCausalLM, "float16")
    assert "fast_dllm.flash_attn_bf16" in mod_c.script()


@pytest.mark.skipif(not _has_metal_codegen, reason="TVM built without Metal codegen")
def test_metal_cross_compile():
    from fast_dllm_relax.targets.metal import metal_target

    mod_n, _ = _export(FastDLLMRelaxForCausalLMNative, "float16", tied=True)
    tvm.compile(mod_n, metal_target(), relax_pipeline=relax.get_pipeline("opt_fast_dllm_llm_native"))


@needs_cuda
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_cuda_pipeline_offloads_to_cublas(dtype):
    import fast_dllm_relax.runtime  # noqa: F401  registers the attention callback

    mod_c, _ = _export(FastDLLMRelaxForCausalLM, dtype, tied=True)
    target = tvm.target.Target.from_device(tvm.cuda(0))
    ex = tvm.compile(
        mod_c, target, relax_pipeline=relax.get_pipeline("opt_fast_dllm_llm_cublas", verbose=False)
    )
    kinds = check_byoc_offload(ex)
    if dtype == "float16":
        # CUTLASS claims the batched fp16 attention matmuls (not the bf16 ones).
        assert kinds.get("static_library", 0) > 0


@needs_cuda
def test_forward_cached_block_matches_forward_cached():
    """forward_cached_block's block K/V are exactly the tail of forward_cached's extended cache."""
    cfg = tiny_config(tied=True)
    dev = tvm.cuda(0)
    _, vm, named = _compiled_cublas(True)
    params = [tvm.runtime.tensor(a, device=dev) for a in _random_params(named).values()]

    NL, NKV, HD, HS = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, cfg.hidden_size
    PLEN, BSIZ = 64, cfg.bd_size
    rng = np.random.default_rng(1)
    t = lambda shape: tvm.runtime.tensor(rng.standard_normal(shape).astype("float16"), dev)  # noqa: E731
    active = t((1, BSIZ, HS))
    ck, cv = t((NL, 1, NKV, PLEN, HD)), t((NL, 1, NKV, PLEN, HD))
    off = tvm.runtime.tensor(np.asarray(PLEN, dtype="int32"), dev)

    h_full, full_k, full_v = (r.numpy() for r in vm["forward_cached"](active, ck, cv, off, params))
    h_blk, blk_k, blk_v = (r.numpy() for r in vm["forward_cached_block"](active, ck, cv, off, params))
    assert blk_k.shape == (NL, 1, NKV, BSIZ, HD)
    np.testing.assert_array_equal(h_blk, h_full)
    np.testing.assert_array_equal(blk_k, full_k[:, :, :, PLEN:, :])
    np.testing.assert_array_equal(blk_v, full_v[:, :, :, PLEN:, :])


@needs_cuda
def test_tied_get_logits_matches_untied():
    """Tied get_logits (embed_tokens.weight) == untied lm_head holding the same array, bit for bit,
    and cuBLAS claims as many kernels either way."""
    dev = tvm.cuda(0)
    ex_t, vm_t, named_t = _compiled_cublas(True)
    ex_u, vm_u, named_u = _compiled_cublas(False)
    assert check_byoc_offload(ex_t).get("cublas_json") == check_byoc_offload(ex_u).get("cublas_json")

    tied = _random_params(named_t)
    untied = dict(tied, **{"lm_head.weight": tied["model.embed_tokens.weight"]})
    params_t = [tvm.runtime.tensor(tied[n], device=dev) for n, _ in named_t]
    params_u = [tvm.runtime.tensor(untied[n], device=dev) for n, _ in named_u]

    rng = np.random.default_rng(2)
    hidden = tvm.runtime.tensor(rng.standard_normal((1, 32, tiny_config().hidden_size)).astype("float16"), dev)
    np.testing.assert_array_equal(
        vm_t["get_logits"](hidden, params_t).numpy(), vm_u["get_logits"](hidden, params_u).numpy()
    )


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

    params = [tvm.runtime.tensor(a, device=dev) for a in _random_params(named).values()]

    NL, NKV, HD, HS = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, cfg.hidden_size
    PLEN, BSIZ, SBS = 64, cfg.bd_size, cfg.bd_size // 4
    rng = np.random.default_rng(0)
    t = lambda shape: tvm.runtime.tensor(rng.standard_normal(shape).astype("float16"), dev)  # noqa: E731
    active, small = t((1, BSIZ, HS)), t((1, SBS, HS))
    ck, cv = t((NL, 1, NKV, PLEN, HD)), t((NL, 1, NKV, PLEN, HD))
    bk, bv = t((NL, 1, NKV, BSIZ, HD)), t((NL, 1, NKV, BSIZ, HD))
    off = lambda v: tvm.runtime.tensor(np.asarray(v, dtype="int32"), dev)  # noqa: E731

    def run(vm):
        outs = [
            vm["forward_cached"](active, ck, cv, off(PLEN), params),
            vm["forward_cached_block"](active, ck, cv, off(PLEN), params),
        ]
        for rp in (0, 8, 16, 24):
            outs.append(vm[f"forward_block_partial_rp{rp}"](small, ck, cv, bk, bv, off(PLEN + rp), params))
        return [[r.numpy().astype("float32") for r in o] for o in outs]

    for outs_n, outs_c in zip(run(vm_n), run(vm_c)):
        for a, b in zip(outs_n, outs_c):
            assert np.isfinite(a).all() and np.isfinite(b).all()
            np.testing.assert_allclose(a, b, rtol=1e-2, atol=1e-2)
