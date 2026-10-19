"""Shared Fast-dLLM runner metrics and result helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


DEFAULT_MODEL_NAME = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"
# HF snapshot all reported numbers were produced with (weights + trust_remote_code modeling code).
DEFAULT_MODEL_REVISION = "25093b6f63300adfd57f72145083c8a528fe4f16"
DEFAULT_PROMPT = "Answer in one short paragraph: What is ONNX Runtime?"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_BASELINE_RESULTS_DIR = "results/fast_dllm_baseline"
DEFAULT_RELAX_RESULTS_DIR = "results/fast_dllm_relax"
DEFAULT_RESULTS_DIR = DEFAULT_BASELINE_RESULTS_DIR
DEFAULT_ARTIFACT_DIR_BF16 = "artifacts/fast_dllm_relax_bf16"
DEFAULT_ARTIFACT_DIR_FP16 = "artifacts/fast_dllm_relax_fp16"
DEFAULT_ARTIFACT_DIR = DEFAULT_ARTIFACT_DIR_BF16
DEFAULT_BLOCK_SIZE = 32
DEFAULT_SMALL_BLOCK_SIZE = 8
DEFAULT_THRESHOLD = 0.9
DEFAULT_TOP_P = 0.95
DEFAULT_TEMPERATURE = 0.0
DEFAULT_USE_BLOCK_CACHE = False
DEFAULT_MASK_ID = 151665
DEFAULT_STOP_TOKEN = 151645

_relax_memory_clear_fn = None
_relax_memory_clear_checked = False


def release_relax_memory_pool() -> None:
    """Best-effort release of TVM's pooled-allocator idle buffers back to the device driver.

    Safe at any point: only touches buffers TVM's allocator has already Free()'d (i.e. already
    fully dereferenced on the Python side), so nothing still in use can be freed. Without it,
    long eval runs with varying sequence lengths accumulate idle pooled buffers until OOM.
    """
    global _relax_memory_clear_fn, _relax_memory_clear_checked
    if not _relax_memory_clear_checked:
        import tvm

        _relax_memory_clear_fn = tvm.get_global_func(
            "vm.builtin.memory_manager.clear", allow_missing=True
        )
        _relax_memory_clear_checked = True
    if _relax_memory_clear_fn is not None:
        _relax_memory_clear_fn()


@dataclass
class GenerationResult:
    """Generated text plus token accounting for one request."""

    prompt: str
    response: str
    generated_ids: Any
    prompt_tokens: int
    generated_tokens: int
    latency_s: float
    generated_tokens_per_s: float
    total_tokens_per_s: float
    max_new_tokens: int
    backend_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestRecord:
    """JSON-serializable per-request generation metrics."""

    request_id: str
    timestamp: str
    model_name: str
    backend: str
    prompt: str
    response: str
    prompt_tokens: int
    generated_tokens: int
    max_new_tokens: int
    latency_s: float
    generated_tokens_per_s: float
    total_tokens_per_s: float
    backend_metrics: dict[str, Any] = field(default_factory=dict)


class ResultRecorder:
    """Shared result collection, summaries, and JSON writing for Fast-dLLM runners."""

    backend: str = "unknown"
    results_dir: str = DEFAULT_RESULTS_DIR

    def __init__(
        self,
        *,
        stream_records: bool = False,
        text_payload_mode: str = "inline",
        preview_chars: int = 160,
    ) -> None:
        if text_payload_mode not in {"inline", "sidecar", "omit"}:
            raise ValueError(
                "text_payload_mode must be one of: inline, sidecar, omit; "
                f"got {text_payload_mode!r}"
            )
        self.records: list[RequestRecord] = []
        self.run_started_at = datetime.now(timezone.utc).isoformat()
        self.load_latency_s: float | None = None
        self.batch_size: int | None = None
        self.stream_records = bool(stream_records)
        self.text_payload_mode = text_payload_mode
        self.preview_chars = max(0, int(preview_chars))
        self._records_file: TextIO | None = None
        self._texts_file: TextIO | None = None
        self._stream_paths: dict[str, Path] = {}
        self._summary_totals = {
            "num_requests": 0,
            "total_latency": 0.0,
            "total_prompt_tokens": 0,
            "total_generated_tokens": 0,
        }
        self._backend_metric_totals: dict[str, float] = {}
        self._backend_metric_counts: dict[str, int] = {}
        self._backend_metric_max: dict[str, float] = {}

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _preview_text(self, text: str) -> str:
        if self.preview_chars <= 0:
            return ""
        return text[: self.preview_chars]

    def _update_aggregates(
        self,
        *,
        prompt_tokens: int,
        generated_tokens: int,
        latency_s: float,
        backend_metrics: dict[str, Any],
    ) -> None:
        self._summary_totals["num_requests"] += 1
        self._summary_totals["total_latency"] += float(latency_s)
        self._summary_totals["total_prompt_tokens"] += int(prompt_tokens)
        self._summary_totals["total_generated_tokens"] += int(generated_tokens)
        for key, value in backend_metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            numeric_value = float(value)
            self._backend_metric_totals[key] = self._backend_metric_totals.get(key, 0.0) + numeric_value
            self._backend_metric_counts[key] = self._backend_metric_counts.get(key, 0) + 1
            self._backend_metric_max[key] = max(
                numeric_value,
                self._backend_metric_max.get(key, numeric_value),
            )

    def start_results(self, results_dir: str | Path | None = None, *, run_name: str = "run") -> dict[str, Path]:
        if self.stream_records and self._records_file is not None:
            return self._stream_paths

        results_path = Path(self.results_dir if results_dir is None else results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        records_path = results_path / f"{run_name}_requests.jsonl"
        summary_path = results_path / f"{run_name}_summary.json"
        paths = {"records": records_path, "summary": summary_path}

        if self.stream_records and self._records_file is None:
            self._records_file = records_path.open("w", encoding="utf-8")
            if self.text_payload_mode == "sidecar":
                texts_path = results_path / f"{run_name}_texts.jsonl"
                self._texts_file = texts_path.open("w", encoding="utf-8")
                paths["texts"] = texts_path

        self._stream_paths = paths
        return paths

    def append_record(
        self,
        *,
        request_id: str | int | None,
        model_name: str,
        prompt: str,
        response: str,
        prompt_tokens: int,
        generated_tokens: int,
        max_new_tokens: int,
        latency_s: float,
        generated_tokens_per_s: float,
        total_tokens_per_s: float,
        backend_metrics: dict[str, Any] | None = None,
    ) -> None:
        backend_metrics = backend_metrics or {}
        request_id_str = str(
            self._summary_totals["num_requests"] if request_id is None else request_id
        )
        timestamp = datetime.now(timezone.utc).isoformat()
        self._update_aggregates(
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            latency_s=latency_s,
            backend_metrics=backend_metrics,
        )

        if self.stream_records:
            if self._records_file is None:
                self.start_results(run_name="run")
            assert self._records_file is not None

            prompt_sha256 = self._hash_text(prompt)
            response_sha256 = self._hash_text(response)
            record: dict[str, Any] = {
                "request_id": request_id_str,
                "timestamp": timestamp,
                "model_name": model_name,
                "backend": self.backend,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "max_new_tokens": max_new_tokens,
                "latency_s": latency_s,
                "generated_tokens_per_s": generated_tokens_per_s,
                "total_tokens_per_s": total_tokens_per_s,
                "backend_metrics": backend_metrics,
                "prompt_sha256": prompt_sha256,
                "response_sha256": response_sha256,
            }
            if self.preview_chars > 0:
                record["prompt_preview"] = self._preview_text(prompt)
                record["response_preview"] = self._preview_text(response)
            if self.text_payload_mode == "inline":
                record["prompt"] = prompt
                record["response"] = response
            elif self.text_payload_mode == "sidecar":
                record["text_payload_ref"] = request_id_str
                if self._texts_file is not None:
                    payload = {
                        "request_id": request_id_str,
                        "prompt": prompt,
                        "response": response,
                        "prompt_sha256": prompt_sha256,
                        "response_sha256": response_sha256,
                    }
                    self._texts_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    self._texts_file.flush()

            self._records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._records_file.flush()
            return

        self.records.append(
            RequestRecord(
                request_id=request_id_str,
                timestamp=timestamp,
                model_name=model_name,
                backend=self.backend,
                prompt=prompt,
                response=response,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                max_new_tokens=max_new_tokens,
                latency_s=latency_s,
                generated_tokens_per_s=generated_tokens_per_s,
                total_tokens_per_s=total_tokens_per_s,
                backend_metrics=backend_metrics,
            )
        )

    def _summary_extra(self) -> dict[str, Any]:
        return {}

    def metric_sum(self, key: str) -> float:
        if self.stream_records:
            return self._backend_metric_totals.get(key, 0.0)
        return sum(
            float(record.backend_metrics.get(key, 0.0))
            for record in self.records
            if isinstance(record.backend_metrics.get(key, 0.0), (int, float))
        )

    def metric_max(self, key: str) -> float | None:
        if self.stream_records:
            return self._backend_metric_max.get(key)
        values = [
            float(record.backend_metrics[key])
            for record in self.records
            if isinstance(record.backend_metrics.get(key), (int, float))
        ]
        return max(values) if values else None

    def metric_avg(self, key: str) -> float:
        if self.stream_records:
            count = self._backend_metric_counts.get(key, 0)
            return self._backend_metric_totals.get(key, 0.0) / count if count else 0.0
        values = [
            float(record.backend_metrics[key])
            for record in self.records
            if isinstance(record.backend_metrics.get(key), (int, float))
        ]
        return sum(values) / len(values) if values else 0.0

    def summary(self) -> dict[str, Any]:
        if self.stream_records:
            total_latency = float(self._summary_totals["total_latency"])
            total_prompt_tokens = int(self._summary_totals["total_prompt_tokens"])
            total_generated_tokens = int(self._summary_totals["total_generated_tokens"])
            num_requests = int(self._summary_totals["num_requests"])
        else:
            total_latency = sum(record.latency_s for record in self.records)
            total_prompt_tokens = sum(record.prompt_tokens for record in self.records)
            total_generated_tokens = sum(record.generated_tokens for record in self.records)
            num_requests = len(self.records)

        summary = {
            "run_started_at": self.run_started_at,
            "run_finished_at": datetime.now(timezone.utc).isoformat(),
            "model_name": getattr(self, "model_name", None),
            "backend": self.backend,
            "batch_size": self.batch_size,
            "num_requests": num_requests,
            "load_latency_s": self.load_latency_s,
            "total_elapsed_s": total_latency,
            "avg_latency_s": total_latency / num_requests if num_requests else 0.0,
            "total_prompt_tokens": total_prompt_tokens,
            "total_generated_tokens": total_generated_tokens,
            "aggregate_generated_tokens_per_s": (
                total_generated_tokens / total_latency if total_latency > 0 else 0.0
            ),
            "aggregate_total_tokens_per_s": (
                (total_prompt_tokens + total_generated_tokens) / total_latency
                if total_latency > 0
                else 0.0
            ),
        }
        summary.update(self._summary_extra())
        return summary

    def finish_results(self) -> None:
        for handle_name in ("_records_file", "_texts_file"):
            handle = getattr(self, handle_name)
            if handle is not None:
                handle.flush()
                handle.close()
                setattr(self, handle_name, None)

    def write_results(
        self,
        results_dir: str | Path | None = None,
        *,
        run_name: str = "run",
    ) -> dict[str, Path]:
        paths = self.start_results(results_dir, run_name=run_name)
        assert "summary" in paths

        if not self.stream_records:
            with paths["records"].open("w", encoding="utf-8") as records_file:
                for record in self.records:
                    records_file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        else:
            self.finish_results()

        with paths["summary"].open("w", encoding="utf-8") as summary_file:
            json.dump(self.summary(), summary_file, indent=2, ensure_ascii=False)
            summary_file.write("\n")

        return paths


def print_generation_result(result: GenerationResult, *, index: int | None = None) -> None:
    label = f"Request {index}" if index is not None else "Request"
    print("=" * 20)
    print(label)
    print("prompt:", result.prompt)
    print("response:", result.response)
    metrics = [
        f"prompt_tokens={result.prompt_tokens}",
        f"generated_tokens={result.generated_tokens}",
        f"latency_s={result.latency_s:.4f}",
        f"generated_tokens_per_s={result.generated_tokens_per_s:.2f}",
    ]
    if result.backend_metrics:
        for key, value in result.backend_metrics.items():
            if isinstance(value, float):
                metrics.append(f"{key}={value:.4f}")
            else:
                metrics.append(f"{key}={value}")
    print("metrics:", *metrics)
    print("=" * 20)
