#!/usr/bin/env python3
"""Time the cached decode building blocks of a CUDA Relax artifact (runtime-level, synced medians).

  python scripts/bench_cached_step.py --artifact-dir artifacts/fast_dllm_relax_bf16 --out results/x.json

Measures, at each prefix length P:
  refine   one refinement full step (embed + cached forward + get_logits + torch reductions)
  append   one block-append (the same, committing the extended cache)
  partial  one 8-token partial forward at replace_position=8 (block-cache path)
and, at the largest P, the prefix-cache build (`build_prefix_cache_from_ids`) and its host-side
attention-bias construction alone. Token ids are random; timings do not depend on them.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from fast_dllm_relax import runtime as rt
from fast_dllm_relax.model.cuda import build_prefix_block_attention_bias
from fast_dllm_relax.runners import FastDLLMRelaxRunner


def _median_ms(fn, *, iters: int, warmup: int, sync) -> float:
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--artifact-dir", default="artifacts/fast_dllm_relax_bf16")
    parser.add_argument("--prefix-lens", default="64,704,2048")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", default=None, help="write the results as JSON here")
    args = parser.parse_args()

    runner = FastDLLMRelaxRunner(artifact_dir=args.artifact_dir, load_model=True)
    vm, params, dev, reshape = runner.vm, runner.params, runner.dev, runner.reshape
    heads = runner.hf_config.num_attention_heads
    bs, sbs = runner.block_size, runner.small_block_size

    def sync() -> None:
        dev.sync()
        torch.cuda.synchronize()

    rng = np.random.default_rng(0)
    prefix_lens = [int(p) for p in args.prefix_lens.split(",")]
    timed = dict(iters=args.iters, warmup=args.warmup, sync=sync)
    results: dict = {"artifact_dir": str(args.artifact_dir), "iters": args.iters, "steps": []}

    for plen in prefix_lens:
        prefix_ids = rng.integers(0, 150000, plen).tolist()
        block_ids = rng.integers(0, 150000, bs).tolist()
        cache = rt.build_prefix_cache_from_ids(
            vm, params, dev, reshape, prefix_ids, num_attention_heads=heads, block_size=bs,
            cache_spec=runner.cache_spec, capacity=plen + bs,
        )

        def refine():
            rt.score_active_block_refine(vm, params, dev, reshape, block_ids, cache)

        def append():
            rt.score_active_block_cached_fast(vm, params, dev, reshape, block_ids, cache)
            cache.prefix_len = plen  # un-commit, so every iteration appends at the same P

        refine()  # the partial step reads the block rows a full step wrote

        def partial():
            rt.score_active_block_partial_perlayer(
                vm, params, dev, reshape, block_ids[8 : 8 + sbs], cache, replace_position=8, block_size=bs
            )

        row = {
            "P": plen,
            "refine_ms": _median_ms(refine, **timed),
            "append_ms": _median_ms(append, **timed),
            "partial_ms": _median_ms(partial, **timed),
        }
        rt.release_relax_memory_pool()
        results["steps"].append(row)
        print(json.dumps(row), flush=True)

    plen = max(prefix_lens)
    prefix_ids = rng.integers(0, 150000, plen).tolist()
    prefill_iters = dict(timed, iters=max(5, args.iters // 5), warmup=2)
    results["prefill"] = {
        "P": plen,
        "prefill_ms": _median_ms(
            lambda: rt.build_prefix_cache_from_ids(
                vm, params, dev, reshape, prefix_ids, num_attention_heads=heads, block_size=bs,
                cache_spec=runner.cache_spec,
            ),
            **prefill_iters,
        ),
        # Exactly the host work build_prefix_cache_from_ids does before its VM call.
        "host_bias_ms": _median_ms(
            lambda: build_prefix_block_attention_bias(prefix_len=0, block_len=plen, block_size=bs),
            **prefill_iters,
        ),
    }
    print(json.dumps(results["prefill"]), flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
