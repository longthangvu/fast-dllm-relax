"""Hugging Face Fast-dLLM loading and prompt helpers."""

from __future__ import annotations

import os
from types import MethodType
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .common import DEFAULT_MASK_ID, DEFAULT_MODEL_REVISION, DEFAULT_STOP_TOKEN


def load_fast_dllm(
    model_name: str,
    *,
    device: str = "cuda:0",
    trust_remote_code: bool = True,
    torch_dtype: str | torch.dtype = "auto",
    revision: str | None = DEFAULT_MODEL_REVISION,
):
    """Load the Fast-dLLM tokenizer and HF model (remote code) at a pinned snapshot revision."""
    token = os.getenv("HF_TOKEN")

    safe_model_name = model_name.replace("/", "__")
    offload_folder = os.path.join(os.path.dirname(__file__), ".offload_fast_dllm", safe_model_name)
    os.makedirs(offload_folder, exist_ok=True)

    config = AutoConfig.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=trust_remote_code,
        revision=revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=trust_remote_code,
        revision=revision,
    )

    model_kwargs: dict[str, Any] = {
        "config": config,
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
        "token": token,
        "revision": revision,
    }
    if device == "auto":
        model_kwargs["device_map"] = "auto"
        model_kwargs["offload_folder"] = offload_folder
    else:
        model_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    # Some checkpoints expose the Fast-dLLM batch sampler as an auxiliary method.
    # Bind it when present so callers can opt in later without hard-failing.
    batch_sample = getattr(model.__class__, "batch_sample", None)
    if callable(batch_sample) and not hasattr(model, "mdm_sample"):
        model.mdm_sample = MethodType(batch_sample, model)

    return tokenizer, model


def build_inputs(tokenizer, prompt: str, device: torch.device | str, already_templated: bool = False):
    """Tokenize a prompt for generation.

    When `already_templated` is True, `prompt` is assumed to already be a
    fully-rendered chat-template string (e.g. lm-eval called the harness's
    own `apply_chat_template()` because `--apply_chat_template` was passed),
    so it's tokenized directly instead of being wrapped in a second user
    turn and re-templated — matching how Fast-dLLM's own official eval
    harness (`generate_until`) consumes already-templated prompts."""
    if already_templated:
        return tokenizer([prompt], return_tensors="pt").to(device)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer([text], return_tensors="pt").to(device)


def resolve_generation_defaults(tokenizer, *, mask_id: int | None = None, stop_token: int | None = None) -> tuple[int | None, int | None]:
    resolved_mask_id = mask_id
    if resolved_mask_id is None:
        resolved_mask_id = getattr(tokenizer, "mask_token_id", None)
    if resolved_mask_id is None:
        resolved_mask_id = DEFAULT_MASK_ID
    resolved_stop_token = stop_token
    if resolved_stop_token is None:
        resolved_stop_token = getattr(tokenizer, "eos_token_id", None)
    if resolved_stop_token is None:
        resolved_stop_token = DEFAULT_STOP_TOKEN
    return resolved_mask_id, resolved_stop_token
