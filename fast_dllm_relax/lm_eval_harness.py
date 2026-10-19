"""lm_eval adapter for the Fast-dLLM baseline and Relax runners.

Run as `python -m fast_dllm_relax.lm_eval_harness <lm_eval args>` (see scripts/eval.sh); importing
this module registers the `fast_dllm_baseline` and `fast_dllm_relax` lm_eval model types.
"""

from __future__ import annotations

from typing import Any

from fast_dllm_relax.common import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_RELAX_RESULTS_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SMALL_BLOCK_SIZE,
    DEFAULT_TEMPERATURE,
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_P,
    DEFAULT_USE_BLOCK_CACHE,
    print_generation_result,
    release_relax_memory_pool,
)
from fast_dllm_relax.runners import FastDLLMBaselineRunner, FastDLLMRelaxRunner

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


def _trim_until(text: str, until: Any) -> str:
    if until is None:
        return text
    stops = until if isinstance(until, (list, tuple)) else [until]
    trimmed = text
    for stop in stops:
        if stop is None:
            continue
        stop = str(stop)
        pos = trimmed.find(stop)
        if pos >= 0:
            trimmed = trimmed[:pos]
    return trimmed


@register_model("fast_dllm_baseline")
class FastDLLMEvalHarness(LM):
    """Generation-only lm_eval harness over the Fast-dLLM baseline runner."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        batch_size: int = 1,
        show_io: bool = True,
        results_dir: str = f"{DEFAULT_RESULTS_DIR}/lm_eval",
        seed: int | None = None,
        device: str = "cuda:0",
        block_size: int = DEFAULT_BLOCK_SIZE,
        small_block_size: int = DEFAULT_SMALL_BLOCK_SIZE,
        threshold: float = DEFAULT_THRESHOLD,
        top_p: float = DEFAULT_TOP_P,
        temperature: float = DEFAULT_TEMPERATURE,
        use_block_cache: bool = DEFAULT_USE_BLOCK_CACHE,
        mask_id: int | None = None,
        stop_token: int | None = None,
        stream_records: bool = False,
        text_payload_mode: str = "inline",
        preview_chars: int = 160,
        already_templated: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.max_new_tokens = int(max_new_tokens)
        self.batch_size = int(batch_size)
        self.show_io = show_io
        self.results_dir = results_dir
        self.seed = seed
        self.device = device
        self.block_size = int(block_size)
        self.small_block_size = int(small_block_size)
        self.threshold = float(threshold)
        self.top_p = float(top_p)
        self.temperature = float(temperature)
        self.use_block_cache = bool(use_block_cache)
        self.mask_id = mask_id
        self.stop_token = stop_token
        self.stream_records = stream_records
        self.text_payload_mode = text_payload_mode
        self.preview_chars = int(preview_chars)
        self.already_templated = already_templated
        self.runner = self._make_runner(**kwargs)
        self.runner.batch_size = self.batch_size
        self._rank = 0
        self._world_size = 1

    def _make_runner(self, **_kwargs):  # type: ignore[no-untyped-def]
        if self.batch_size != 1:
            raise ValueError("Fast-dLLM lm_eval currently supports only batch_size=1.")
        return FastDLLMBaselineRunner(
            self.model_name,
            max_new_tokens=self.max_new_tokens,
            seed=self.seed,
            device=self.device,
            block_size=self.block_size,
            small_block_size=self.small_block_size,
            threshold=self.threshold,
            top_p=self.top_p,
            temperature=self.temperature,
            use_block_cache=self.use_block_cache,
            mask_id=self.mask_id,
            stop_token=self.stop_token,
            stream_records=self.stream_records,
            text_payload_mode=self.text_payload_mode,
            preview_chars=self.preview_chars,
            already_templated=self.already_templated,
        )

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        return self.model_name

    def loglikelihood_rolling(self, requests):  # type: ignore[no-untyped-def]
        raise NotImplementedError("generation tasks only")

    def loglikelihood(self, requests):  # type: ignore[no-untyped-def]
        results: list[tuple[float, bool]] = []
        for request in requests:
            context, continuation = request.args
            result = self.runner.loglikelihood(context, continuation)
            if getattr(self.runner, "backend", None) == "tvm_relax":
                release_relax_memory_pool()
            results.append(result)
            if self.show_io:
                print(
                    f"[loglikelihood] ctx=...{context[-80:]!r} cont={continuation!r} "
                    f"-> logprob={result[0]:.4f} is_greedy={result[1]}"
                )
        return results

    def apply_chat_template(self, chat_history, add_generation_prompt: bool = True):  # type: ignore[no-untyped-def]
        self.runner.load()
        return self.runner.tokenizer.apply_chat_template(
            chat_history,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )

    def _runner_generation_kwargs(self, gen_kwargs: Any) -> dict[str, Any]:
        if not isinstance(gen_kwargs, dict):
            return {}
        generation_kwargs = dict(gen_kwargs)
        generation_kwargs.pop("until", None)
        generation_kwargs.pop("max_gen_toks", None)
        return generation_kwargs

    def generate_until(self, requests):  # type: ignore[no-untyped-def]
        outputs: list[str | None] = [None] * len(requests)
        if self.stream_records:
            self.runner.start_results(self.results_dir, run_name="lm_eval")

        for index, request in enumerate(requests):
            prompt = request.args[0]
            gen_kwargs = request.args[1] if len(request.args) > 1 else {}
            until = gen_kwargs.get("until") if isinstance(gen_kwargs, dict) else None
            max_gen_toks = (
                gen_kwargs.get("max_gen_toks", self.max_new_tokens)
                if isinstance(gen_kwargs, dict)
                else self.max_new_tokens
            )
            max_new_tokens = max(self.max_new_tokens, int(max_gen_toks))

            result = self.runner.generate(
                str(prompt),
                request_id=index,
                max_new_tokens=max_new_tokens,
                generation_kwargs=self._runner_generation_kwargs(gen_kwargs),
            )
            if getattr(self.runner, "backend", None) == "tvm_relax":
                release_relax_memory_pool()
            response = _trim_until(result.response, until)
            outputs[index] = response

            if self.show_io:
                print_generation_result(result, index=index + 1)

        self.runner.write_results(self.results_dir, run_name="lm_eval")
        return outputs


_RELAX_RUNNER_KWARGS = frozenset(
    {"block_size", "small_block_size", "threshold", "top_p", "temperature", "use_block_cache", "max_new_tokens"}
)


@register_model("fast_dllm_relax")
class FastDLLMRelaxEvalHarness(FastDLLMEvalHarness):
    """Generation-only lm_eval harness over the Fast-dLLM Relax/TVM runner."""

    def __init__(
        self,
        artifact_dir: str = DEFAULT_ARTIFACT_DIR,
        results_dir: str = f"{DEFAULT_RELAX_RESULTS_DIR}/lm_eval",
        chunked_prefill: bool = False,
        **kwargs,
    ) -> None:
        self.artifact_dir = artifact_dir
        self.chunked_prefill = bool(chunked_prefill)
        super().__init__(results_dir=results_dir, **kwargs)

    def _runner_generation_kwargs(self, gen_kwargs: Any) -> dict[str, Any]:
        kwargs = super()._runner_generation_kwargs(gen_kwargs)
        return {k: v for k, v in kwargs.items() if k in _RELAX_RUNNER_KWARGS}

    def _make_runner(self, **_kwargs):  # type: ignore[no-untyped-def]
        if self.batch_size != 1:
            raise ValueError("Fast-dLLM lm_eval currently supports only batch_size=1.")
        return FastDLLMRelaxRunner(
            self.model_name,
            artifact_dir=self.artifact_dir,
            max_new_tokens=self.max_new_tokens,
            seed=self.seed,
            device=self.device,
            block_size=self.block_size,
            small_block_size=self.small_block_size,
            threshold=self.threshold,
            top_p=self.top_p,
            temperature=self.temperature,
            use_block_cache=self.use_block_cache,
            mask_id=self.mask_id,
            stop_token=self.stop_token,
            stream_records=self.stream_records,
            text_payload_mode=self.text_payload_mode,
            preview_chars=self.preview_chars,
            already_templated=self.already_templated,
            chunked_prefill=self.chunked_prefill,
        )


if __name__ == "__main__":
    cli_evaluate()
