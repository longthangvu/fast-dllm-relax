"""Tiny-config compile checks for both Relax models (random weights, no checkpoint download).

    pytest tests/          # with the TVM env from scripts/setup_tvm.sh

CUDA tests are skipped without a CUDA device. The Metal-model tests are skipped until the Metal
model is ported to the in-place KV cache (plan 003); before that, the Metal cross-compile test was
skipped when this TVM build has no Metal codegen, and the static-vs-dynamic test ran on CUDA or
Metal, whichever existed.
"""

import functools

import numpy as np
import pytest
import torch
import tvm
from tvm import relax

from fast_dllm_relax.export_utils import REQUIRED_FNS, check_byoc_offload
from fast_dllm_relax.model.cuda import (
    FastDLLMRelaxConfig,
    FastDLLMRelaxForCausalLM,
    build_prefix_block_attention_bias,
)
from fast_dllm_relax.model.metal import METAL_STATIC_FNS, FastDLLMRelaxForCausalLMNative, metal_tir_pipeline

_has_cuda = tvm.cuda(0).exist
_has_metal_codegen = tvm.get_global_func("target.build.metal", allow_missing=True) is not None
needs_cuda = pytest.mark.skipif(not _has_cuda, reason="no CUDA device")
metal_not_ported = pytest.mark.skip(reason="Metal model not ported to the in-place KV cache (plan 003)")


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
def test_exported_functions_and_param_lists(tied):
    mod_c, named_c = _export(FastDLLMRelaxForCausalLM, "float16", tied=tied)
    names = {gv.name_hint for gv in mod_c.get_global_vars()}
    assert set(REQUIRED_FNS) <= names
    # Static-shape variants are Metal-only; the CUDA artifact keeps exactly the dynamic set.
    assert not set(METAL_STATIC_FNS) & names
    assert "forward_cached_block" not in names  # merged into forward_cached (in-place KV cache)
    # A tied model reuses embed_tokens.weight in get_logits instead of carrying lm_head.weight.
    assert ("lm_head.weight" in dict(named_c)) == (not tied)
    assert "model.embed_tokens.weight" in dict(named_c)


@metal_not_ported
@pytest.mark.parametrize("tied", [False, True])
def test_native_exports_same_functions_and_params(tied):
    mod_c, named_c = _export(FastDLLMRelaxForCausalLM, "float16", tied=tied)
    mod_n, named_n = _export(FastDLLMRelaxForCausalLMNative, "float16", tied=tied)
    assert set(REQUIRED_FNS) <= {gv.name_hint for gv in mod_n.get_global_vars()}
    assert set(METAL_STATIC_FNS) <= {gv.name_hint for gv in mod_n.get_global_vars()}
    # Same names/order/shapes, so one params.npz layout serves both models.
    sig = lambda named: [(n, [int(d) for d in p.shape], str(p.dtype)) for n, p in named]  # noqa: E731
    assert sig(named_c) == sig(named_n)


@metal_not_ported
def test_native_model_has_no_attention_callback():
    mod_n, _ = _export(FastDLLMRelaxForCausalLMNative, "float16")
    assert "flash_attn" not in mod_n.script()
    mod_c, _ = _export(FastDLLMRelaxForCausalLM, "float16")
    assert "fast_dllm.attn_inplace" in mod_c.script()


@metal_not_ported
@pytest.mark.skipif(not _has_metal_codegen, reason="TVM built without Metal codegen")
@pytest.mark.parametrize("narrow_i32", [False, True])
def test_metal_cross_compile(narrow_i32, tmp_path):
    from fast_dllm_relax.targets.metal import inspect_metal_tar, metal_target

    mod_n, _ = _export(FastDLLMRelaxForCausalLMNative, "float16", tied=True)
    ex = tvm.compile(
        mod_n,
        metal_target(),
        relax_pipeline=relax.get_pipeline("opt_fast_dllm_llm_native"),
        tir_pipeline=metal_tir_pipeline(narrow_i32),
    )
    ex.export_library(str(tmp_path / "tiny.tar"))
    info = inspect_metal_tar(tmp_path / "tiny.tar")
    if narrow_i32:
        assert info["long_count"] == 0
    else:
        assert info["long_count"] > 0  # symbolic-shape kernels index in int64 without narrowing


def _static_test_device():
    if _has_cuda:
        return tvm.cuda(0)
    if tvm.metal(0).exist:
        return tvm.metal(0)
    return None


@metal_not_ported
@pytest.mark.skipif(_static_test_device() is None, reason="no CUDA or Metal device")
def test_static_variants_match_dynamic():
    """Each static-shape function computes what its dynamic sibling computes, at the static size.

    Runs on whichever GPU is present, with the Metal pipeline (int32-narrowed on Metal, as exported).
    """
    cfg = tiny_config(tied=True)
    dev = _static_test_device()
    on_metal = dev.dlpack_device_type() == tvm.metal(0).dlpack_device_type()
    if on_metal:
        from fast_dllm_relax.targets.metal import metal_target

        target = metal_target()
    else:
        target = tvm.target.Target.from_device(dev)
    mod_n, named = _export(FastDLLMRelaxForCausalLMNative, "float16", tied=True)
    ex = tvm.compile(
        mod_n,
        target,
        relax_pipeline=relax.get_pipeline("opt_fast_dllm_llm_native"),
        tir_pipeline=metal_tir_pipeline(on_metal),
    )
    vm = relax.VirtualMachine(ex, dev)
    params = [tvm.runtime.tensor(a, device=dev) for a in _random_params(named).values()]

    NL, NKV, HD, HS = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, cfg.hidden_size
    PLEN, BSIZ, SBS = 64, cfg.bd_size, cfg.bd_size // 4
    rng = np.random.default_rng(3)
    t = lambda shape: tvm.runtime.tensor(rng.standard_normal(shape).astype("float16"), dev)  # noqa: E731
    active = t((1, BSIZ, HS))
    ck, cv = t((NL, 1, NKV, PLEN, HD)), t((NL, 1, NKV, PLEN, HD))
    off = tvm.runtime.tensor(np.asarray(PLEN, dtype="int32"), dev)

    pairs = [
        ("forward_cached_b32", "forward_cached", (active, ck, cv, off)),
        ("forward_cached_block_b32", "forward_cached_block", (active, ck, cv, off)),
        ("get_logits_m32", "get_logits", (t((1, BSIZ, HS)),)),
        ("get_logits_m8", "get_logits", (t((1, SBS, HS)),)),
    ]
    as_list = lambda out: [out] if hasattr(out, "numpy") else list(out)  # noqa: E731  tensor or tuple
    for static, dynamic, args in pairs:
        outs_s, outs_d = as_list(vm[static](*args, params)), as_list(vm[dynamic](*args, params))
        assert len(outs_s) == len(outs_d)
        for a, b in zip(outs_s, outs_d):
            a, b = a.numpy().astype("float32"), b.numpy().astype("float32")
            assert a.shape == b.shape and np.isfinite(a).all()
            if on_metal:  # the shipped configuration: bit-exact (observed on an M4)
                np.testing.assert_array_equal(a, b, err_msg=f"{static} vs {dynamic}")
            else:
                np.testing.assert_allclose(a, b, rtol=5e-3, atol=5e-3, err_msg=f"{static} vs {dynamic}")


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


def _tiny_cuda_setup(seed: int):
    """(cfg, dev, vm, params, rand) for the tiny tied fp16 CUDA model; rand(shape) -> fp16 TVM tensor."""
    cfg = tiny_config(tied=True)
    dev = tvm.cuda(0)
    _, vm, named = _compiled_cublas(True)
    params = [tvm.runtime.tensor(a, device=dev) for a in _random_params(named).values()]
    rng = np.random.default_rng(seed)
    rand = lambda shape: tvm.runtime.tensor(rng.standard_normal(shape).astype("float16"), dev)  # noqa: E731
    return cfg, dev, vm, params, rand


def _torch_cache(cfg, cap: int, fill: float = 7.0) -> tuple[torch.Tensor, torch.Tensor]:
    """A sentinel-filled in-place KV cache pair, as torch tensors (zero-copy TVM views via from_dlpack)."""
    shape = (cfg.num_hidden_layers, 1, cap, cfg.num_key_value_heads, cfg.head_dim)
    return (torch.full(shape, fill, dtype=torch.float16, device="cuda"),
            torch.full(shape, fill, dtype=torch.float16, device="cuda"))


def _tv(t: torch.Tensor):
    return tvm.runtime.from_dlpack(t)


def _off(dev, v: int):
    return tvm.runtime.tensor(np.asarray(v, dtype="int32"), dev)


def _prefill(vm, params, dev, embed, k, v):
    plen = int(embed.shape[1])
    bias = tvm.runtime.tensor(build_prefix_block_attention_bias(0, plen, block_size=32), dev)
    vm["build_prefix_cache"](embed, bias, _tv(k), _tv(v), params)
    dev.sync()


@needs_cuda
def test_inplace_cache_write_windows():
    """Each compiled cache call writes exactly its own row window of the in-place KV cache."""
    cfg, dev, vm, params, rand = _tiny_cuda_setup(1)
    HS, BSIZ, SBS, PLEN, CAP = cfg.hidden_size, cfg.bd_size, cfg.bd_size // 4, 32, 96
    k, v = _torch_cache(cfg, CAP)
    prefix, block = rand((1, PLEN, HS)), rand((1, BSIZ, HS))

    _prefill(vm, params, dev, prefix, k, v)
    for c in (k, v):
        assert not (c[:, :, :PLEN] == 7.0).any(), "build_prefix_cache left prefix rows unwritten"
        assert (c[:, :, PLEN:] == 7.0).all(), "build_prefix_cache wrote past the prefix"

    k_prefix = k[:, :, :PLEN].clone()
    h = vm["forward_cached"](block, _tv(k), _tv(v), _off(dev, PLEN), tvm.runtime.ShapeTuple([PLEN]), params)
    dev.sync()
    assert tuple(h.shape) == (1, BSIZ, HS)
    assert torch.equal(k[:, :, :PLEN], k_prefix), "forward_cached modified the committed prefix"
    assert not (k[:, :, PLEN:PLEN + BSIZ] == 7.0).any(), "forward_cached left block rows unwritten"
    assert (k[:, :, PLEN + BSIZ:] == 7.0).all(), "forward_cached wrote past the block"

    # The block rows hold the same K as a dense prefill of prefix + block at those rows (layer 0's
    # K depends only on the row's embedding and position; GEMM shapes differ, so not bit-exact).
    k_dense, v_dense = _torch_cache(cfg, CAP)
    _prefill(vm, params, dev, tvm.runtime.tensor(np.concatenate([prefix.numpy(), block.numpy()], axis=1), dev),
             k_dense, v_dense)
    torch.testing.assert_close(k[0, :, PLEN:PLEN + BSIZ], k_dense[0, :, PLEN:PLEN + BSIZ], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v[0, :, PLEN:PLEN + BSIZ], v_dense[0, :, PLEN:PLEN + BSIZ], rtol=1e-2, atol=1e-2)

    for rp in (0, 8, 16, 24):
        window = slice(PLEN + rp, PLEN + rp + SBS)
        k[:, :, window] = 7.0
        before = k.clone()
        vm[f"forward_block_partial_rp{rp}"](
            rand((1, SBS, HS)), _tv(k), _tv(v), _off(dev, PLEN + rp), tvm.runtime.ShapeTuple([PLEN]), params
        )
        dev.sync()
        outside = torch.ones(CAP, dtype=torch.bool, device="cuda")
        outside[window] = False
        assert torch.equal(k[:, :, outside], before[:, :, outside]), f"rp{rp} wrote outside its window"
        assert not (k[:, :, window] == 7.0).any(), f"rp{rp} left its window unwritten"


@needs_cuda
def test_inplace_matches_dense_reference():
    """Cached forward == the dense block-causal forward_hidden on the same rows; a partial forward of
    a sub-block == the full cached step's rows for it (same tokens). Different attention kernels
    (in-graph fp32 vs the torch callback), hence the fp16 tolerance."""
    cfg, dev, vm, params, rand = _tiny_cuda_setup(2)
    HS, BSIZ, SBS, PLEN = cfg.hidden_size, cfg.bd_size, cfg.bd_size // 4, 64
    seq = rand((1, PLEN + BSIZ, HS)).numpy()
    bias = tvm.runtime.tensor(build_prefix_block_attention_bias(0, PLEN + BSIZ, block_size=BSIZ), dev)
    h_dense = vm["forward_hidden"](tvm.runtime.tensor(seq, dev), bias, params).numpy()[:, PLEN:]

    k, v = _torch_cache(cfg, PLEN + BSIZ)
    _prefill(vm, params, dev, tvm.runtime.tensor(np.ascontiguousarray(seq[:, :PLEN]), dev), k, v)
    plen = tvm.runtime.ShapeTuple([PLEN])
    h_cached = vm["forward_cached"](
        tvm.runtime.tensor(np.ascontiguousarray(seq[:, PLEN:]), dev), _tv(k), _tv(v), _off(dev, PLEN), plen, params
    ).numpy()
    np.testing.assert_allclose(h_cached.astype("float32"), h_dense.astype("float32"), rtol=2e-2, atol=2e-2)

    for rp in (0, 8, 16, 24):
        small = tvm.runtime.tensor(np.ascontiguousarray(seq[:, PLEN + rp:PLEN + rp + SBS]), dev)
        h_part = vm[f"forward_block_partial_rp{rp}"](small, _tv(k), _tv(v), _off(dev, PLEN + rp), plen, params).numpy()
        np.testing.assert_allclose(
            h_part.astype("float32"), h_cached[:, rp:rp + SBS].astype("float32"), rtol=2e-2, atol=2e-2,
            err_msg=f"rp{rp}",
        )


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


@metal_not_ported
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
