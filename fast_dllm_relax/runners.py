"""Fast-dLLM generation runners: Hugging Face baseline and compiled Relax artifact."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import ml_dtypes  # noqa: F401 — registers bfloat16 dtype with numpy (needed for any ml_dtypes arrays)
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from .common import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    DEFAULT_PROMPT,
    DEFAULT_RELAX_RESULTS_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SMALL_BLOCK_SIZE,
    DEFAULT_TEMPERATURE,
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_P,
    DEFAULT_USE_BLOCK_CACHE,
    GenerationResult,
    ResultRecorder,
)
from .hf_generation import build_inputs, load_fast_dllm, resolve_generation_defaults
from .metrics import (
    compact_optional_metrics,
    estimate_effective_bandwidth_gbps,
    estimate_npz_param_bytes,
    estimate_torch_model_bytes,
    reset_torch_cuda_peak_stats,
    snapshot_process_gpu_memory,
    snapshot_torch_cuda,
    snapshot_tvm_available_memory,
)

tvm = None
relax_greedy_generate = None
score_masked_continuation = None
selected_fn_names = None


def _ensure_tvm_imported() -> None:
    """Import TVM (and the TVM-dependent Relax runtime) lazily so baseline-only
    usage never requires a compiled TVM library."""
    global tvm, relax_greedy_generate, score_masked_continuation, selected_fn_names
    if tvm is not None:
        return
    import tvm as _tvm
    from .runtime import relax_greedy_generate as _relax_greedy_generate
    from .runtime import score_masked_continuation as _score_masked_continuation
    from .runtime import selected_fn_names as _selected_fn_names

    tvm = _tvm
    relax_greedy_generate = _relax_greedy_generate
    score_masked_continuation = _score_masked_continuation
    selected_fn_names = _selected_fn_names


def _make_tvm_param(arr, artifact_dtype: str, dev):
    """Create a TVM tensor from a loaded numpy param array.

    bfloat16 artifacts store params as uint16 (portable npz); reconstruct by
    allocating a bfloat16 TVM buffer and copyfrom the raw uint16 bits.
    """
    if artifact_dtype == "bfloat16":
        return tvm.runtime.empty(arr.shape, "bfloat16", dev).copyfrom(arr)
    return tvm.runtime.tensor(arr, device=dev)


def _max_present(*values: Any) -> int | None:
    numeric_values = [int(value) for value in values if isinstance(value, (int, float)) and value > 0]
    return max(numeric_values) if numeric_values else None


def _encode_context_continuation(tokenizer, context: str, continuation: str) -> tuple[list[int], list[int]]:
    """Split (context, continuation) into token id lists via a tokenization diff.

    Mirrors the official Fast-dLLM eval harness's `_encode_pair`
    (Fast-dLLM/v2/eval.py:161-168): tokenizing context and continuation
    separately can produce different subword boundaries than tokenizing them
    jointly (e.g. merging the context's last character with the
    continuation's first). Tokenizing the concatenation once and slicing off
    the context's own token count avoids that drift.
    """
    whole_ids = tokenizer(context + continuation, add_special_tokens=False)["input_ids"]
    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    return context_ids, whole_ids[len(context_ids) :]


def warmup_fast_dllm(
    model,
    tokenizer,
    *,
    max_new_tokens: int = 64,
    block_size: int = DEFAULT_BLOCK_SIZE,
    small_block_size: int = DEFAULT_SMALL_BLOCK_SIZE,
    threshold: float = DEFAULT_THRESHOLD,
    top_p: float = DEFAULT_TOP_P,
    temperature: float = DEFAULT_TEMPERATURE,
    use_block_cache: bool = DEFAULT_USE_BLOCK_CACHE,
    mask_id: int | None = None,
    stop_token: int | None = None,
):
    resolved_mask_id, resolved_stop_token = resolve_generation_defaults(
        tokenizer,
        mask_id=mask_id,
        stop_token=stop_token,
    )
    model_inputs = build_inputs(tokenizer, DEFAULT_PROMPT, model.device)

    warmup_kwargs = {
        "tokenizer": tokenizer,
        "max_new_tokens": int(max_new_tokens),
        "block_size": int(block_size),
        "small_block_size": int(small_block_size),
        "threshold": float(threshold),
        "top_p": float(top_p),
        "temperature": float(temperature),
        "use_block_cache": bool(use_block_cache),
    }
    if resolved_mask_id is not None:
        warmup_kwargs["mask_id"] = int(resolved_mask_id)
    if resolved_stop_token is not None:
        warmup_kwargs["stop_token"] = int(resolved_stop_token)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    _ = model.generate(model_inputs["input_ids"], **warmup_kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return time.perf_counter() - start


class FastDLLMBaselineRunner(ResultRecorder):
    """Reusable Fast-dLLM Hugging Face baseline runner."""

    backend = "huggingface_fast_dllm"
    results_dir = DEFAULT_RESULTS_DIR

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        seed: int | None = None,
        load_model: bool = True,
        device: str = "cuda:0",
        block_size: int = DEFAULT_BLOCK_SIZE,
        small_block_size: int = DEFAULT_SMALL_BLOCK_SIZE,
        threshold: float = DEFAULT_THRESHOLD,
        top_p: float = DEFAULT_TOP_P,
        temperature: float = DEFAULT_TEMPERATURE,
        use_block_cache: bool = DEFAULT_USE_BLOCK_CACHE,
        trust_remote_code: bool = True,
        torch_dtype: str | torch.dtype = "auto",
        mask_id: int | None = None,
        stop_token: int | None = None,
        stream_records: bool = False,
        text_payload_mode: str = "inline",
        preview_chars: int = 160,
        already_templated: bool = False,
        revision: str | None = DEFAULT_MODEL_REVISION,
    ) -> None:
        super().__init__(
            stream_records=stream_records,
            text_payload_mode=text_payload_mode,
            preview_chars=preview_chars,
        )
        self.model_name = model_name
        self.max_new_tokens = int(max_new_tokens)
        self.seed = seed
        self.device = device
        self.block_size = int(block_size)
        self.small_block_size = int(small_block_size)
        self.threshold = float(threshold)
        self.top_p = float(top_p)
        self.temperature = float(temperature)
        self.use_block_cache = bool(use_block_cache)
        self.trust_remote_code = bool(trust_remote_code)
        self.torch_dtype = torch_dtype
        self.revision = revision
        self.mask_id = mask_id
        self.stop_token = stop_token
        self.already_templated = bool(already_templated)
        self.tokenizer = None
        self.model = None
        self.estimated_model_bytes: int | None = None
        self.load_memory_metrics: dict[str, Any] = {}

        if load_model:
            self.load()

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        start = time.perf_counter()
        load_memory_before = snapshot_torch_cuda(torch, prefix="memory_before")
        load_process_before = snapshot_process_gpu_memory(prefix="process_gpu_memory_before")
        self.tokenizer, self.model = load_fast_dllm(
            self.model_name,
            device=self.device,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=self.torch_dtype,
            revision=self.revision,
        )
        warmup_latency_s = warmup_fast_dllm(self.model, self.tokenizer)
        print(f"Warmup latency:{warmup_latency_s}")
        self.mask_id, self.stop_token = resolve_generation_defaults(
            self.tokenizer,
            mask_id=self.mask_id,
            stop_token=self.stop_token,
        )
        self.estimated_model_bytes = estimate_torch_model_bytes(self.model)
        load_memory_after = snapshot_torch_cuda(torch, prefix="memory_after")
        load_process_after = snapshot_process_gpu_memory(prefix="process_gpu_memory_after")
        self.load_memory_metrics = {
            **load_memory_before,
            **load_memory_after,
            **load_process_before,
            **load_process_after,
        }
        self.load_latency_s = time.perf_counter() - start

    def generate(
        self,
        prompt: str,
        *,
        request_id: str | int | None = None,
        max_new_tokens: int | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        record: bool = True,
    ) -> GenerationResult:
        self.load()
        assert self.tokenizer is not None
        assert self.model is not None

        generation_kwargs = dict(generation_kwargs or {})
        kwarg_max_new_tokens = generation_kwargs.pop("max_new_tokens", None)
        if max_new_tokens is None and kwarg_max_new_tokens is not None:
            max_new_tokens = int(kwarg_max_new_tokens)
        max_new_tokens = self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)

        model_inputs = build_inputs(
            self.tokenizer, prompt, self.model.device, already_templated=self.already_templated
        )
        prompt_token_count = int(model_inputs.input_ids.shape[-1])

        effective_kwargs = {
            "tokenizer": self.tokenizer,
            "block_size": int(generation_kwargs.pop("block_size", self.block_size)),
            "small_block_size": int(generation_kwargs.pop("small_block_size", self.small_block_size)),
            "threshold": float(generation_kwargs.pop("threshold", self.threshold)),
            "top_p": float(generation_kwargs.pop("top_p", self.top_p)),
            "temperature": float(generation_kwargs.pop("temperature", self.temperature)),
            "use_block_cache": bool(generation_kwargs.pop("use_block_cache", self.use_block_cache)),
            "max_new_tokens": max_new_tokens,
        }
        if self.mask_id is not None:
            effective_kwargs["mask_id"] = int(generation_kwargs.pop("mask_id", self.mask_id))
        if self.stop_token is not None:
            effective_kwargs["stop_token"] = int(generation_kwargs.pop("stop_token", self.stop_token))
        effective_kwargs.update(generation_kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        memory_before = snapshot_torch_cuda(torch, prefix="memory_before")
        process_before = snapshot_process_gpu_memory(prefix="process_gpu_memory_before")
        reset_torch_cuda_peak_stats(torch)
        start = time.perf_counter()

        # The model's generate() is third-party block-diffusion code (loaded via
        # trust_remote_code) with no prefill/decode split of its own — every
        # forward pass, prefill or decode, goes through `self(...)`, i.e. real
        # `nn.Module.__call__`, so a forward hook sees them all. Its first call
        # processes the (near-)full prompt in one shot (see modeling.py's
        # `generate()`: `if input_ids.shape[1] > block_size: output = self(...)`
        # before the block loop starts) — true prefill whenever the prompt
        # exceeds one block (block_size, default 32), which holds for every
        # prompt this project actually runs. Only that boundary is timed (one
        # extra sync, not one per forward call).
        hook_target = self.model
        prefill_timing: dict[str, float] = {}

        def _mark_prefill_boundary(module, args, kwargs, output):
            if "prefill_latency_s" in prefill_timing:
                return
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            prefill_timing["prefill_latency_s"] = time.perf_counter() - start
            hook_handle.remove()

        hook_handle = hook_target.register_forward_hook(_mark_prefill_boundary, with_kwargs=True)
        try:
            generated_ids = self.model.generate(model_inputs["input_ids"], **effective_kwargs)
        finally:
            hook_handle.remove()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        memory_after = snapshot_torch_cuda(torch, prefix="memory_after")
        process_after = snapshot_process_gpu_memory(prefix="process_gpu_memory_after")
        latency_s = time.perf_counter() - start
        prefill_latency_s = float(prefill_timing.get("prefill_latency_s", 0.0))
        decode_latency_s = max(latency_s - prefill_latency_s, 0.0)

        output_ids = generated_ids[0][prompt_token_count:].tolist()
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        generated_token_count = len(output_ids)
        generated_tokens_per_s = generated_token_count / latency_s if latency_s > 0 else 0.0
        total_tokens_per_s = (
            (prompt_token_count + generated_token_count) / latency_s if latency_s > 0 else 0.0
        )
        decode_tokens_per_s = generated_token_count / decode_latency_s if decode_latency_s > 0 else 0.0
        peak_metrics = snapshot_torch_cuda(torch, prefix="memory")
        backend_metrics = compact_optional_metrics(
            {
                **memory_before,
                **memory_after,
                **peak_metrics,
                **process_before,
                **process_after,
                "process_gpu_memory_peak_bytes": _max_present(
                    process_before.get("process_gpu_memory_before_peak_bytes"),
                    process_after.get("process_gpu_memory_after_peak_bytes"),
                ),
                "estimated_model_bytes": self.estimated_model_bytes,
                "estimated_weight_bandwidth_gbps": estimate_effective_bandwidth_gbps(
                    self.estimated_model_bytes,
                    generated_token_count,
                    latency_s,
                ),
                "estimated_total_token_bandwidth_gbps": estimate_effective_bandwidth_gbps(
                    self.estimated_model_bytes,
                    prompt_token_count + generated_token_count,
                    latency_s,
                ),
                "block_size": effective_kwargs["block_size"],
                "small_block_size": effective_kwargs["small_block_size"],
                "threshold": effective_kwargs["threshold"],
                "top_p": effective_kwargs["top_p"],
                "temperature": effective_kwargs["temperature"],
                "use_block_cache": effective_kwargs["use_block_cache"],
                "mask_id": effective_kwargs.get("mask_id"),
                "stop_token": effective_kwargs.get("stop_token"),
                "prefill_latency_s": prefill_latency_s,
                "decode_latency_s": decode_latency_s,
                "decode_tokens_per_s": decode_tokens_per_s,
            }
        )

        result = GenerationResult(
            prompt=prompt,
            response=response,
            generated_ids=generated_ids,
            prompt_tokens=prompt_token_count,
            generated_tokens=generated_token_count,
            latency_s=latency_s,
            generated_tokens_per_s=generated_tokens_per_s,
            total_tokens_per_s=total_tokens_per_s,
            max_new_tokens=max_new_tokens,
            backend_metrics=backend_metrics,
        )

        if record:
            self.append_record(
                request_id=request_id,
                model_name=self.model_name,
                prompt=prompt,
                response=response,
                prompt_tokens=prompt_token_count,
                generated_tokens=generated_token_count,
                max_new_tokens=max_new_tokens,
                latency_s=latency_s,
                generated_tokens_per_s=generated_tokens_per_s,
                total_tokens_per_s=total_tokens_per_s,
                backend_metrics=backend_metrics,
            )

        return result

    def loglikelihood(self, context: str, continuation: str) -> tuple[float, bool]:
        """Diffusion loglikelihood: mask the whole continuation, one forward pass.

        Generalizes the official Fast-dLLM eval harness's
        `_forward_process`/`get_loglikelihood` (Fast-dLLM/v2/eval.py:171-206),
        which only ever masks the continuation's *first* token (fine for
        MMLU's single-token "A"/"B"/"C"/"D" choices, silently incomplete for
        GPQA's 2-token "(A)"-style choices). Masking the entire continuation
        span instead reduces to the official repo's exact behavior when the
        continuation is one token, and correctly scores multi-token
        continuations too — still one forward pass, not one per token.

        Block-diffusion attention is bidirectional *within* a block
        (see `build_prefix_block_attention_bias` in model/cuda.py), so
        feeding the real continuation tokens through unmasked would let each
        position's prediction see its own (and later) answer tokens — masking
        is what prevents that leak, not an optional detail.
        """
        self.load()
        assert self.tokenizer is not None
        assert self.model is not None
        assert self.mask_id is not None

        context_ids, continuation_ids = _encode_context_continuation(self.tokenizer, context, continuation)
        if not continuation_ids:
            return 0.0, True

        seq = context_ids + continuation_ids
        seq_len = len(seq)
        mask_start = len(context_ids)

        perturbed = list(seq)
        for index in range(mask_start, seq_len):
            perturbed[index] = self.mask_id

        pad_len = -seq_len % self.block_size
        if pad_len:
            perturbed.extend([self.mask_id] * pad_len)
            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id is not None:
                perturbed[seq_len] = eos_token_id

        input_ids = torch.tensor([perturbed], device=self.model.device)
        with torch.no_grad():
            logits = self.model(input_ids).logits
        # Shift-and-realign to match the official repo's `get_logits`
        # (Fast-dLLM/v2/eval.py:182-186): shifted[:, i, :] becomes logits
        # from position i-1, so it lines up with scoring the token AT
        # position i (row 0 is left as-is — there is no position -1).
        shifted_logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

        log_probs = torch.log_softmax(shifted_logits[0, mask_start:seq_len, :].float(), dim=-1)
        target = torch.tensor(continuation_ids, device=self.model.device)
        token_log_probs = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        return float(token_log_probs.sum().item()), False

    def _summary_extra(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "generation_config": {
                "max_new_tokens": self.max_new_tokens,
                "seed": self.seed,
                "device": self.device,
                "block_size": self.block_size,
                "small_block_size": self.small_block_size,
                "threshold": self.threshold,
                "top_p": self.top_p,
                "temperature": self.temperature,
                "use_block_cache": self.use_block_cache,
                "mask_id": self.mask_id,
                "stop_token": self.stop_token,
            },
            "memory_breakdown": compact_optional_metrics(
                {
                    "estimated_model_bytes": self.estimated_model_bytes,
                    "max_memory_peak_allocated_bytes": self.metric_max("memory_peak_allocated_bytes"),
                    "max_memory_peak_reserved_bytes": self.metric_max("memory_peak_reserved_bytes"),
                    "max_process_gpu_memory_peak_bytes": self.metric_max("process_gpu_memory_peak_bytes"),
                    "avg_estimated_weight_bandwidth_gbps": self.metric_avg("estimated_weight_bandwidth_gbps"),
                    "max_estimated_weight_bandwidth_gbps": self.metric_max("estimated_weight_bandwidth_gbps"),
                }
            ),
            "load_memory_metrics": self.load_memory_metrics,
            "memory_notes": (
                "PyTorch CUDA memory fields are allocator observations. "
                "Process GPU memory uses optional NVML when available. "
                "Bandwidth fields are lower-bound estimates, not hardware counters."
            ),
        }


class FastDLLMRelaxRunner(ResultRecorder):
    """Reusable Fast-dLLM runner backed by exported Relax artifacts."""

    backend = "tvm_relax"
    results_dir = DEFAULT_RELAX_RESULTS_DIR

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        seed: int | None = None,
        load_model: bool = True,
        device: str = "cuda:0",
        block_size: int = DEFAULT_BLOCK_SIZE,
        small_block_size: int = DEFAULT_SMALL_BLOCK_SIZE,
        threshold: float = DEFAULT_THRESHOLD,
        top_p: float = DEFAULT_TOP_P,
        temperature: float = DEFAULT_TEMPERATURE,
        use_block_cache: bool = DEFAULT_USE_BLOCK_CACHE,
        trust_remote_code: bool = True,
        mask_id: int | None = None,
        stop_token: int | None = None,
        stream_records: bool = False,
        text_payload_mode: str = "inline",
        preview_chars: int = 160,
        already_templated: bool = False,
        chunked_prefill: bool = False,
        revision: str | None = DEFAULT_MODEL_REVISION,
    ) -> None:
        _ensure_tvm_imported()
        super().__init__(
            stream_records=stream_records,
            text_payload_mode=text_payload_mode,
            preview_chars=preview_chars,
        )
        self.model_name = model_name
        self.artifact_dir = Path(artifact_dir)
        self.max_new_tokens = int(max_new_tokens)
        self.seed = seed
        self.device = device
        self.block_size = int(block_size)
        self.small_block_size = int(small_block_size)
        self.threshold = float(threshold)
        self.top_p = float(top_p)
        self.temperature = float(temperature)
        self.use_block_cache = bool(use_block_cache)
        self.trust_remote_code = bool(trust_remote_code)
        self.mask_id = mask_id
        self.stop_token = stop_token
        self.already_templated = bool(already_templated)
        # Chunked prefix-cache construction (see runtime.build_prefix_cache_from_ids).
        self.chunked_prefill = bool(chunked_prefill)
        self.revision = revision

        self.tokenizer = None
        self.hf_config = None
        self.loaded_ex = None
        self.vm = None
        self.params = None
        self.dev = None
        self.reshape = None
        self.param_names: list[str] = []
        self.artifact_file_bytes: dict[str, int] = {}
        self.estimated_param_bytes: int | None = None
        self.load_memory_metrics: dict[str, Any] = {}
        # Cross-request prefix-cache memo for loglikelihood(): MMLU/GPQA
        # issue 4 requests/question sharing the identical context, so consecutive
        # calls with the same block-aligned prefix reuse the built KV cache instead
        # of rebuilding it from scratch each time. See score_masked_continuation's
        # `cached_prefix` parameter.
        self._loglikelihood_prefix_cache: tuple[list[int], Any] | None = None
        # Total-length cap from the artifact manifest (int32-indexed Metal artifacts); None = no cap.
        self.max_total_len: int | None = None

        if load_model:
            self.load()

    def load(self) -> None:
        if self.vm is not None and self.tokenizer is not None and self.params is not None:
            return

        if self.seed is not None:
            np.random.seed(self.seed)

        start = time.perf_counter()
        token = os.getenv("HF_TOKEN")
        self.hf_config = AutoConfig.from_pretrained(
            self.model_name,
            token=token,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=token,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
        )
        self.mask_id, self.stop_token = resolve_generation_defaults(
            self.tokenizer,
            mask_id=self.mask_id,
            stop_token=self.stop_token,
        )

        library_path, params_path, param_names_path = self._artifact_paths()
        missing = [path for path in (library_path, params_path, param_names_path) if not path.exists()]
        if missing:
            formatted = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(f"Missing Relax artifact file(s):\n{formatted}")

        self.dev = self._make_device(self.device)
        load_available_before = snapshot_tvm_available_memory(
            self.dev,
            prefix="available_global_memory_before",
        )
        load_process_before = snapshot_process_gpu_memory(prefix="process_gpu_memory_before")
        self.artifact_file_bytes = {
            "library": library_path.stat().st_size,
            "params": params_path.stat().st_size,
            "param_names": param_names_path.stat().st_size,
        }
        self.loaded_ex = tvm.runtime.load_module(str(library_path))
        self.vm = tvm.relax.VirtualMachine(self.loaded_ex, self.dev)
        _CACHE_REQUIRED_GLOBALS = (
            "build_prefix_cache", "forward_cached", "forward_cached_block", "get_logits",
        )
        for _fn_name in _CACHE_REQUIRED_GLOBALS:
            try:
                self.vm[_fn_name]
            except Exception:
                raise RuntimeError(
                    f"Relax artifact is missing cache-aware global '{_fn_name}'. "
                    "Re-export the artifact with scripts/export_cuda.py (or scripts/export_metal.py)."
                ) from None
        try:
            self.vm["forward_block_partial_rp0"]
            self._has_block_cache = True
        except Exception:
            self._has_block_cache = False
        # Metal artifacts built with int32 device indexing record a total-length cap; CUDA/Jetson
        # artifacts have no manifest, hence no cap.
        manifest_path = self.artifact_dir / "fast_dllm_relax_nn_metal_manifest.json"
        if manifest_path.exists():
            self.max_total_len = json.loads(manifest_path.read_text(encoding="utf-8")).get("max_total_len")
        self.param_names = param_names_path.read_text(encoding="utf-8").splitlines()
        self.params = self._load_params(params_path, self.param_names)
        self.reshape = tvm.get_global_func("vm.builtin.reshape")
        load_available_after = snapshot_tvm_available_memory(
            self.dev,
            prefix="available_global_memory_after",
        )
        load_process_after = snapshot_process_gpu_memory(prefix="process_gpu_memory_after")
        self.load_memory_metrics = {
            **load_available_before,
            **load_available_after,
            **load_process_before,
            **load_process_after,
        }
        self.load_latency_s = time.perf_counter() - start

    def generate(
        self,
        prompt: str,
        *,
        request_id: str | int | None = None,
        max_new_tokens: int | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        record: bool = True,
    ) -> GenerationResult:
        self.load()
        assert self.tokenizer is not None
        assert self.hf_config is not None
        assert self.vm is not None
        assert self.params is not None
        assert self.dev is not None
        assert self.reshape is not None

        generation_kwargs = dict(generation_kwargs or {})
        kwarg_max_new_tokens = generation_kwargs.pop("max_new_tokens", None)
        if max_new_tokens is None and kwarg_max_new_tokens is not None:
            max_new_tokens = int(kwarg_max_new_tokens)
        max_new_tokens = self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)

        effective_kwargs = {
            "block_size": int(generation_kwargs.pop("block_size", self.block_size)),
            "small_block_size": int(generation_kwargs.pop("small_block_size", self.small_block_size)),
            "threshold": float(generation_kwargs.pop("threshold", self.threshold)),
            "top_p": float(generation_kwargs.pop("top_p", self.top_p)),
            "temperature": float(generation_kwargs.pop("temperature", self.temperature)),
            "use_block_cache": bool(generation_kwargs.pop("use_block_cache", self.use_block_cache)),
            "max_new_tokens": max_new_tokens,
        }
        if generation_kwargs:
            unsupported = ", ".join(sorted(generation_kwargs))
            raise ValueError(f"Unsupported Fast-dLLM Relax generation kwargs: {unsupported}")
        if self.mask_id is None:
            raise ValueError("Fast-dLLM Relax runner requires a resolved mask_id.")

        prompt_ids = self._build_prompt_ids(prompt)
        prompt_token_count = len(prompt_ids)
        self._check_total_len(prompt_token_count + max_new_tokens + effective_kwargs["block_size"], "generate")

        self._sync()
        available_before = snapshot_tvm_available_memory(
            self.dev,
            prefix="available_global_memory_before",
        )
        process_before = snapshot_process_gpu_memory(prefix="process_gpu_memory_before")
        start = time.perf_counter()
        effective_block_cache = effective_kwargs["use_block_cache"]
        if effective_block_cache and not self._has_block_cache:
            import warnings
            warnings.warn(
                "use_block_cache=True requested but artifact lacks forward_block_partial_rp* globals. "
                "Falling back to use_block_cache=False. Re-export with scripts/export_cuda.py.",
                stacklevel=2,
            )
            effective_block_cache = False
        timing: dict[str, Any] = {}
        generated, runtime_stats = relax_greedy_generate(
            self.vm,
            self.params,
            self.dev,
            self.reshape,
            self.tokenizer,
            prompt_ids,
            num_attention_heads=int(self.hf_config.num_attention_heads),
            max_new_tokens=max_new_tokens,
            block_size=effective_kwargs["block_size"],
            small_block_size=effective_kwargs["small_block_size"],
            threshold=effective_kwargs["threshold"],
            temperature=effective_kwargs["temperature"],
            top_p=effective_kwargs["top_p"],
            use_block_cache=effective_block_cache,
            mask_id=int(self.mask_id),
            stop_token=None if self.stop_token is None else int(self.stop_token),
            timing=timing,
            chunked_prefill=self.chunked_prefill,
        )
        self._sync()
        latency_s = time.perf_counter() - start
        available_after = snapshot_tvm_available_memory(
            self.dev,
            prefix="available_global_memory_after",
        )
        process_after = snapshot_process_gpu_memory(prefix="process_gpu_memory_after")

        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        generated_token_count = len(generated)
        generated_tokens_per_s = generated_token_count / latency_s if latency_s > 0 else 0.0
        total_tokens_per_s = (
            (prompt_token_count + generated_token_count) / latency_s if latency_s > 0 else 0.0
        )
        prefill_latency_s = float(timing.get("prefill_latency_s", 0.0))
        decode_latency_s = max(latency_s - prefill_latency_s, 0.0)
        decode_tokens_per_s = generated_token_count / decode_latency_s if decode_latency_s > 0 else 0.0
        backend_metrics = compact_optional_metrics(
            {
                **available_before,
                **available_after,
                **process_before,
                **process_after,
                "process_gpu_memory_peak_bytes": _max_present(
                    process_before.get("process_gpu_memory_before_peak_bytes"),
                    process_after.get("process_gpu_memory_after_peak_bytes"),
                ),
                "estimated_param_bytes": self.estimated_param_bytes,
                "estimated_weight_bandwidth_gbps": estimate_effective_bandwidth_gbps(
                    self.estimated_param_bytes,
                    generated_token_count,
                    latency_s,
                ),
                "block_size": effective_kwargs["block_size"],
                "small_block_size": effective_kwargs["small_block_size"],
                "threshold": effective_kwargs["threshold"],
                "top_p": effective_kwargs["top_p"],
                "temperature": effective_kwargs["temperature"],
                "use_block_cache": effective_kwargs["use_block_cache"],
                "chunked_prefill": self.chunked_prefill,
                **selected_fn_names(
                    self.vm,
                    block_size=effective_kwargs["block_size"],
                    small_block_size=effective_kwargs["small_block_size"],
                ),
                "max_total_len": self.max_total_len,
                "mask_id": self.mask_id,
                "stop_token": self.stop_token,
                "prefill_latency_s": prefill_latency_s,
                "decode_latency_s": decode_latency_s,
                "decode_tokens_per_s": decode_tokens_per_s,
                **runtime_stats,
            }
        )

        result = GenerationResult(
            prompt=prompt,
            response=response,
            generated_ids=generated,
            prompt_tokens=prompt_token_count,
            generated_tokens=generated_token_count,
            latency_s=latency_s,
            generated_tokens_per_s=generated_tokens_per_s,
            total_tokens_per_s=total_tokens_per_s,
            max_new_tokens=max_new_tokens,
            backend_metrics=backend_metrics,
        )

        if record:
            self.append_record(
                request_id=request_id,
                model_name=self.model_name,
                prompt=prompt,
                response=response,
                prompt_tokens=prompt_token_count,
                generated_tokens=generated_token_count,
                max_new_tokens=max_new_tokens,
                latency_s=latency_s,
                generated_tokens_per_s=generated_tokens_per_s,
                total_tokens_per_s=total_tokens_per_s,
                backend_metrics=backend_metrics,
            )

        return result

    def loglikelihood(self, context: str, continuation: str) -> tuple[float, bool]:
        """Diffusion loglikelihood via the Relax backend's `score_masked_continuation`.

        Same masking design as `FastDLLMBaselineRunner.loglikelihood` (mask
        the whole continuation, one forward pass) built on
        `runtime.score_masked_continuation` (prefix KV cache + one masked
        `forward_cached_block` call on the tail block) — no loglikelihood-specific
        compiled function is needed.
        """
        self.load()
        assert self.tokenizer is not None
        assert self.hf_config is not None
        assert self.vm is not None
        assert self.params is not None
        assert self.dev is not None
        assert self.reshape is not None
        assert self.mask_id is not None

        context_ids, continuation_ids = _encode_context_continuation(self.tokenizer, context, continuation)
        if not continuation_ids:
            return 0.0, True
        self._check_total_len(len(context_ids) + len(continuation_ids) + self.block_size, "loglikelihood")

        logits, self._loglikelihood_prefix_cache = score_masked_continuation(
            self.vm,
            self.params,
            self.dev,
            self.reshape,
            context_ids,
            continuation_ids,
            num_attention_heads=int(self.hf_config.num_attention_heads),
            block_size=self.block_size,
            mask_id=int(self.mask_id),
            eos_token_id=self.tokenizer.eos_token_id,
            cached_prefix=self._loglikelihood_prefix_cache,
            chunked_prefill=self.chunked_prefill,
        )
        logits_tensor = torch.from_numpy(np.asarray(logits, dtype=np.float32))
        log_probs = torch.log_softmax(logits_tensor, dim=-1)
        target = torch.tensor(continuation_ids)
        token_log_probs = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        return float(token_log_probs.sum().item()), False

    def _summary_extra(self) -> dict[str, Any]:
        library_path, params_path, param_names_path = self._artifact_paths()
        return {
            "artifact_dir": str(self.artifact_dir),
            "artifact_files": {
                "library": str(library_path),
                "params": str(params_path),
                "param_names": str(param_names_path),
            },
            "artifact_file_bytes": self.artifact_file_bytes,
            "device": self.device,
            "generation_config": {
                "max_new_tokens": self.max_new_tokens,
                "seed": self.seed,
                "block_size": self.block_size,
                "small_block_size": self.small_block_size,
                "threshold": self.threshold,
                "top_p": self.top_p,
                "temperature": self.temperature,
                "use_block_cache": self.use_block_cache,
                "mask_id": self.mask_id,
                "stop_token": self.stop_token,
            },
            "compiled_fns": (
                selected_fn_names(self.vm, block_size=self.block_size, small_block_size=self.small_block_size)
                if self.vm is not None
                else None
            ),
            "max_total_len": self.max_total_len,
            "memory_breakdown": compact_optional_metrics(
                {
                    "estimated_param_bytes": self.estimated_param_bytes,
                    "artifact_total_bytes": sum(self.artifact_file_bytes.values()),
                    "max_process_gpu_memory_peak_bytes": self.metric_max(
                        "process_gpu_memory_peak_bytes"
                    ),
                    "avg_estimated_weight_bandwidth_gbps": self.metric_avg(
                        "estimated_weight_bandwidth_gbps"
                    ),
                }
            ),
            "load_memory_metrics": self.load_memory_metrics,
            "memory_notes": (
                "Relax available-global-memory fields are device-reported when supported. "
                "Process GPU memory uses optional NVML when available. "
                "Bandwidth fields are lower-bound estimates, not hardware counters."
            ),
        }

    def _artifact_paths(self) -> tuple[Path, Path, Path]:
        return (
            self.artifact_dir / "fast_dllm_relax_nn.so",
            self.artifact_dir / "fast_dllm_relax_nn_params.npz",
            self.artifact_dir / "fast_dllm_relax_nn_param_names.txt",
        )

    def _check_total_len(self, total_len: int, where: str) -> None:
        if self.max_total_len is not None and total_len > self.max_total_len:
            raise ValueError(
                f"{where}: total length {total_len} (prompt + new tokens + block padding) exceeds "
                f"max_total_len={self.max_total_len} of this artifact, whose Metal kernels index in "
                "int32 and would overflow silently. Re-export with --no-index-narrow-i32 for longer inputs."
            )

    def _load_params(self, params_path: Path, param_names: list[str]) -> list[Any]:
        dtype_path = self.artifact_dir / "fast_dllm_relax_nn_dtype.txt"
        artifact_dtype = dtype_path.read_text().strip() if dtype_path.exists() else "float16"

        with np.load(params_path) as loaded_param_arrays:
            missing = [
                f"arr_{index}"
                for index in range(len(param_names))
                if f"arr_{index}" not in loaded_param_arrays
            ]
            if missing:
                formatted = ", ".join(missing[:10])
                suffix = "..." if len(missing) > 10 else ""
                raise ValueError(f"Parameter archive is missing expected arrays: {formatted}{suffix}")

            self.estimated_param_bytes = estimate_npz_param_bytes(loaded_param_arrays, param_names)
            params = [
                _make_tvm_param(loaded_param_arrays[f"arr_{index}"], artifact_dtype, self.dev)
                for index, _ in enumerate(param_names)
            ]

        if len(params) != len(param_names):
            raise ValueError(
                f"Loaded {len(params)} params but found {len(param_names)} parameter names."
            )
        return params

    @staticmethod
    def _make_device(device: str) -> Any:
        if device.startswith("cuda"):
            if ":" in device:
                _, index = device.split(":", 1)
                return tvm.device("cuda", int(index))
            return tvm.device("cuda", 0)
        if device.startswith("cpu"):
            if ":" in device:
                _, index = device.split(":", 1)
                return tvm.device("cpu", int(index))
            return tvm.device("cpu", 0)
        if device.startswith("metal"):  # Metal artifact (scripts/export_metal.py)
            if ":" in device:
                _, index = device.split(":", 1)
                return tvm.device("metal", int(index))
            return tvm.device("metal", 0)
        raise ValueError(
            f"Unsupported TVM device string for Fast-dLLM Relax runner: {device} (expected cuda[:N], cpu[:N] or metal[:N])"
        )

    def _build_prompt_ids(self, prompt: str) -> list[int]:
        assert self.tokenizer is not None
        if self.already_templated:
            return list(self.tokenizer(prompt)["input_ids"])
        messages = [{"role": "user", "content": prompt}]
        return list(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )

    def _sync(self) -> None:
        if self.dev is None:
            return
        sync = getattr(self.dev, "sync", None)
        if callable(sync):
            sync()
