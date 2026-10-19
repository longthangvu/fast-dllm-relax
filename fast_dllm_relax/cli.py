"""Terminal demo / quick benchmark CLI for the Fast-dLLM baseline and Relax runners.

Example: python -m fast_dllm_relax.cli --backend relax --use-block-cache --prompt "..."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fast_dllm_relax.common import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_BASELINE_RESULTS_DIR,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_PROMPT,
    DEFAULT_RELAX_RESULTS_DIR,
    DEFAULT_SMALL_BLOCK_SIZE,
    DEFAULT_TEMPERATURE,
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_P,
    print_generation_result,
)
from fast_dllm_relax.runners import FastDLLMBaselineRunner, FastDLLMRelaxRunner


def _load_prompts(path: str | None, prompt: str | None) -> list[str]:
    if path is None:
        return [prompt or DEFAULT_PROMPT]

    prompts: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            item = json.loads(line)
            value = item.get("prompt") or item.get("question") or item.get("input")
            if value is None:
                raise ValueError(f"JSONL prompt row has no prompt/question/input field: {line}")
            prompts.append(str(value))
        else:
            prompts.append(line)

    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("baseline", "relax"), default="baseline")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--run-name", default="demo")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--show-io", default=True)
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--small-block-size", type=int, default=DEFAULT_SMALL_BLOCK_SIZE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--use-block-cache", action="store_true")
    parser.add_argument("--chunked-prefill", action="store_true",
                        help="Relax only: build the prefix KV cache block by block (recommended on Jetson)")
    parser.add_argument("--mask-id", type=int, default=None)
    parser.add_argument("--stop-token", type=int, default=None)
    parser.add_argument("--stream-records", action="store_true")
    parser.add_argument(
        "--text-payload-mode",
        choices=("inline", "sidecar", "omit"),
        default="inline",
    )
    parser.add_argument("--preview-chars", type=int, default=160)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    prompts = _load_prompts(args.prompts_file, args.prompt)
    max_new_tokens = args.max_new_tokens or DEFAULT_MAX_NEW_TOKENS
    if args.backend == "baseline":
        results_dir = args.results_dir or f"{DEFAULT_BASELINE_RESULTS_DIR}/demo"
        runner = FastDLLMBaselineRunner(
            args.model_name,
            max_new_tokens=max_new_tokens,
            seed=args.seed,
            device=args.device,
            block_size=args.block_size,
            small_block_size=args.small_block_size,
            threshold=args.threshold,
            top_p=args.top_p,
            temperature=args.temperature,
            use_block_cache=args.use_block_cache,
            mask_id=args.mask_id,
            stop_token=args.stop_token,
            stream_records=args.stream_records,
            text_payload_mode=args.text_payload_mode,
            preview_chars=args.preview_chars,
        )
    else:
        results_dir = args.results_dir or f"{DEFAULT_RELAX_RESULTS_DIR}/demo"
        runner = FastDLLMRelaxRunner(
            args.model_name,
            artifact_dir=args.artifact_dir,
            max_new_tokens=max_new_tokens,
            seed=args.seed,
            device=args.device,
            block_size=args.block_size,
            small_block_size=args.small_block_size,
            threshold=args.threshold,
            top_p=args.top_p,
            temperature=args.temperature,
            use_block_cache=args.use_block_cache,
            mask_id=args.mask_id,
            stop_token=args.stop_token,
            stream_records=args.stream_records,
            text_payload_mode=args.text_payload_mode,
            preview_chars=args.preview_chars,
            chunked_prefill=args.chunked_prefill,
        )

    if args.stream_records:
        runner.start_results(results_dir, run_name=args.run_name)

    for index, prompt in enumerate(prompts, start=1):
        result = runner.generate(prompt, request_id=index)
        if args.show_io:
            print_generation_result(result, index=index)

    paths = runner.write_results(results_dir, run_name=args.run_name)
    summary = runner.summary()
    print("Summary:")
    print(f"  backend: {summary['backend']}")
    print(f"  requests: {summary['num_requests']}")
    print(f"  avg_latency_s: {summary['avg_latency_s']:.4f}")
    print(
        "  aggregate_generated_tokens_per_s: "
        f"{summary['aggregate_generated_tokens_per_s']:.2f}"
    )
    print(f"  records: {paths['records']}")
    print(f"  summary: {paths['summary']}")


if __name__ == "__main__":
    main()
