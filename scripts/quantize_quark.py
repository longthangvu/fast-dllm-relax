"""Quantize Fast-dLLM v2 1.5B with AMD Quark (weight-only) and export a Quark safetensors checkpoint.

Calibration reflects block-diffusion inference: chat-formatted GSM8K-train prompts+answers (--calib
mix: half of them replaced by wikitext-2 text), packed to a multiple of 32 tokens, with the trailing
block(s) partially replaced by the mask token at random ratios, so the activation statistics include
masked positions as seen during decoding.

Needs a GPU build of PyTorch and `pip install amd-quark` (validated with amd-quark 0.13 on ROCm 7.2).

  # int4 MLP, attention left fp16 (the ROCm mixed-precision artifact; see README)
  python scripts/quantize_quark.py --scheme int4_wo_128 --algo gptq --calib mix --nsamples 256 \
      --exclude "*q_proj,*k_proj,*v_proj,*o_proj" --out quant/fdllm_q4gptq_mlp
  # all projections int4
  python scripts/quantize_quark.py --scheme int4_wo_128 --algo gptq --calib mix --nsamples 256 --out quant/fdllm_q4gptq
  # then: python scripts/export_rocm_quant.py --preset int4-mlp --quark quant/fdllm_q4gptq_mlp

The int4 schemes are symmetric (zero point 0), as scripts/export_rocm_quant.py expects.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fast_dllm_relax.common import DEFAULT_BLOCK_SIZE as BD  # noqa: E402
from fast_dllm_relax.common import DEFAULT_MASK_ID as MASK_ID  # noqa: E402
from fast_dllm_relax.common import DEFAULT_MODEL_NAME as MODEL  # noqa: E402
from fast_dllm_relax.common import DEFAULT_MODEL_REVISION as REV  # noqa: E402


def _mask_tail(ids, g, max_len):
    for b in range(int(torch.randint(1, 3, (1,), generator=g))):
        lo = max_len - (b + 1) * BD
        ratio = float(torch.rand((), generator=g)) * 0.9 + 0.1
        sel = torch.rand(BD, generator=g) < ratio
        ids[lo:lo + BD][sel] = MASK_ID
    return ids


def calib_batches_mix(tok, n: int, max_len: int, seed: int = 0):
    """Half GSM8K chat (as calib_batches), half general text (wikitext-2 train), packed to max_len,
    tail blocks masked."""
    g = torch.Generator().manual_seed(seed + 1)
    half = calib_batches(tok, n // 2, max_len, seed)
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    stream, out = [], []
    for row in wt:
        if len(row["text"].strip()) < 40:
            continue
        stream += tok(row["text"], add_special_tokens=False)["input_ids"]
        while len(stream) >= max_len and len(out) < n - len(half):
            ids = torch.tensor(stream[:max_len], dtype=torch.long); stream = stream[max_len:]
            out.append({"input_ids": _mask_tail(ids, g, max_len).unsqueeze(0)})
        if len(out) >= n - len(half):
            break
    mixed = [None] * (len(half) + len(out))
    mixed[0::2], mixed[1::2] = half[: len(mixed[0::2])], out[: len(mixed[1::2])]
    return mixed


def calib_batches(tok, n: int, max_len: int, seed: int = 0):
    """n fixed-length (max_len, a multiple of 32) sequences: chat-formatted GSM8K-train Q+A examples
    packed back to back, then the last 1-2 blocks partially replaced by the mask token. Fixed length
    keeps layer-wise algorithms (AWQ/GPTQ) consistent: they cache per-layer inputs, incl. RoPE."""
    assert max_len % BD == 0
    g = torch.Generator().manual_seed(seed)
    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed)
    stream, out, i = [], [], 0
    while len(out) < n:
        ex = ds[i % len(ds)]; i += 1
        msgs = [{"role": "user", "content": ex["question"]}, {"role": "assistant", "content": ex["answer"]}]
        stream += tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
        while len(stream) >= max_len and len(out) < n:
            ids = torch.tensor(stream[:max_len], dtype=torch.long); stream = stream[max_len:]
            for b in range(int(torch.randint(1, 3, (1,), generator=g))):
                lo = max_len - (b + 1) * BD
                ratio = float(torch.rand((), generator=g)) * 0.9 + 0.1
                sel = torch.rand(BD, generator=g) < ratio
                ids[lo:lo + BD][sel] = MASK_ID
            out.append({"input_ids": ids.unsqueeze(0)})
    return out


def patch_quark_awq_mask() -> None:
    """Fast-dLLM layers take a 2-D block-diffusion mask [L, L] shared by the whole batch. Quark's AWQ
    helper treats mask dim 0 as the batch and rejects it; leave 2-D (broadcastable) masks as they are."""
    import quark.torch.algorithm.awq.auto_smooth as aws
    import quark.torch.algorithm.awq.awq as awq
    import quark.torch.algorithm.awq.utils as awq_utils

    orig = awq_utils.align_attention_mask_with_input

    def align(module, kwargs, inp=None):
        mask = kwargs.get("attention_mask")
        if mask is not None and mask.dim() == 2:
            return orig(module, kwargs, None)  # still filters kwargs; skips the batch alignment
        return orig(module, kwargs, inp)

    for mod in (awq_utils, awq, aws):
        mod.align_attention_mask_with_input = align


def build_config(scheme: str, algo: str, extra_exclude=()):
    from quark.torch import LLMTemplate
    from quark.torch.quantization.config.config import Int8PerChannelSpec, QConfig, QLayerConfig

    if scheme == "int8_wo":
        spec = Int8PerChannelSpec(ch_axis=0, symmetric=True)
        spec = spec.to_quantization_spec()
        return QConfig(global_quant_config=QLayerConfig(weight=spec), exclude=["lm_head", *extra_exclude])
    tmpl = LLMTemplate.get("qwen2")
    return tmpl.get_config(scheme=scheme, algorithm=None if algo == "none" else algo,
                           exclude_layers=["lm_head", *extra_exclude])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="int4_wo_128")
    ap.add_argument("--algo", default="none", choices=["none", "awq", "gptq"])
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--calib", default="gsm8k", choices=["gsm8k", "mix"],
                    help="gsm8k: chat GSM8K-train only; mix: half GSM8K chat, half wikitext-2")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--exclude", default="", help="comma-separated layer patterns left unquantized (plus lm_head)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dequant-out", default=None,
                    help="also save an fp16 HF checkpoint whose Linear weights are the dequantized (QDQ) "
                         "Quark weights: numerically W4A16/W8A16, but loadable by any fp16 pipeline")
    args = ap.parse_args()
    from quark.torch import ModelQuantizer, export_safetensors

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16).cuda().eval()
    model.config.model_type = "qwen2"  # Quark templates key on model_type; Fast_dLLM_Qwen is Qwen2 layer-wise
    make = calib_batches_mix if args.calib == "mix" else calib_batches
    calib = [{k: v.cuda() for k, v in b.items()} for b in make(tok, args.nsamples, args.max_len)]
    patch_quark_awq_mask()
    cfg = build_config(args.scheme, args.algo, [e for e in args.exclude.split(",") if e])
    print(f"config built ({args.scheme}, {args.algo}); calibrating on {len(calib)} samples ...", flush=True)
    model = ModelQuantizer(cfg).quantize_model(model, calib)
    print(f"quantized in {time.time() - t0:.1f}s", flush=True)

    if args.dequant_out:
        # Every parameter of the quantized model goes into the fp16 checkpoint: QDQ weights for the
        # quantized Linears, and the other parameters as Quark left them. AWQ/SmoothQuant fold their
        # per-channel scales into the preceding RMSNorm / v_proj weights, so copying only the Linear
        # weights (first version) dropped those and broke AWQ models.
        qparams = {}
        with torch.no_grad():
            for name, mod in model.named_modules():
                if hasattr(mod, "get_quant_weight") and getattr(mod, "weight", None) is not None:
                    qparams[name + ".weight"] = mod.get_quant_weight(mod.weight).detach().to(torch.float16).cpu()
            for name, p in model.named_parameters():
                if name not in qparams and "quantizer" not in name:
                    qparams[name] = p.detach().to(torch.float16).cpu()
        ref = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, trust_remote_code=True, torch_dtype=torch.float16)
        sd = ref.state_dict()
        changed, missing = 0, []
        for k, w in qparams.items():
            if k not in sd:
                missing.append(k); continue
            if w.shape != sd[k].shape:
                raise RuntimeError(f"shape mismatch for {k}: {tuple(w.shape)} vs {tuple(sd[k].shape)}")
            if not torch.equal(w, sd[k]):
                changed += 1
            sd[k].copy_(w)
        dq = Path(args.dequant_out); dq.mkdir(parents=True, exist_ok=True)
        ref.save_pretrained(dq, safe_serialization=True)
        tok.save_pretrained(dq)
        print(f"dequantized fp16 checkpoint: {changed} tensors differ from the original "
              f"(of {len(qparams)} copied; {len(missing)} names not in the HF model: {missing[:4]}) -> {dq}", flush=True)
        del ref

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.config.model_type = "Fast_dLLM_Qwen"
    with torch.no_grad():
        export_safetensors(model, out)
    tok.save_pretrained(out)
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(MODEL, revision=REV, allow_patterns=["*.py", "generation_config.json"]))
    for f in snap.glob("*.py"):
        shutil.copy(f, out / f.name)
    if (snap / "generation_config.json").exists():
        shutil.copy(snap / "generation_config.json", out / "generation_config.json")
    print(f"exported to {out} in {time.time() - t0:.1f}s total", flush=True)


if __name__ == "__main__":
    main()
