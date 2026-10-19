"""Best-effort memory and bandwidth helpers for Fast-dLLM runners."""

from __future__ import annotations

import os
from typing import Any


def snapshot_torch_cuda(torch_module: Any, *, prefix: str = "memory") -> dict[str, int]:
    """Return PyTorch CUDA allocator stats when CUDA is available."""

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)):
        return {}
    if not cuda.is_available():
        return {}

    try:
        return {
            f"{prefix}_allocated_bytes": int(cuda.memory_allocated()),
            f"{prefix}_reserved_bytes": int(cuda.memory_reserved()),
            f"{prefix}_peak_allocated_bytes": int(cuda.max_memory_allocated()),
            f"{prefix}_peak_reserved_bytes": int(cuda.max_memory_reserved()),
        }
    except Exception:
        return {}


def reset_torch_cuda_peak_stats(torch_module: Any) -> None:
    """Reset PyTorch CUDA peak memory stats if supported."""

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)):
        return
    if not cuda.is_available():
        return
    reset = getattr(cuda, "reset_peak_memory_stats", None)
    if callable(reset):
        try:
            reset()
        except Exception:
            return


def snapshot_tvm_available_memory(device: Any, *, prefix: str = "available_global_memory") -> dict[str, int]:
    """Return TVM device available memory if the device exposes it."""

    try:
        value = getattr(device, "available_global_memory", None)
    except Exception:
        return {}
    if value is None:
        return {}
    try:
        return {f"{prefix}_bytes": int(value)}
    except (TypeError, ValueError):
        return {}


def snapshot_process_gpu_memory(*, prefix: str = "process_gpu_memory") -> dict[str, int]:
    """Return current process GPU memory via NVML when pynvml is installed."""

    try:
        import pynvml  # type: ignore[import-not-found]
    except ImportError:
        return {}

    try:
        pynvml.nvmlInit()
        pid = os.getpid()
        peak_used = 0
        current_used = 0
        device_count = pynvml.nvmlDeviceGetCount()
        for index in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            processes = []
            for getter_name in (
                "nvmlDeviceGetComputeRunningProcesses",
                "nvmlDeviceGetGraphicsRunningProcesses",
            ):
                getter = getattr(pynvml, getter_name, None)
                if callable(getter):
                    try:
                        processes.extend(getter(handle))
                    except Exception:
                        pass
            for process in processes:
                if int(getattr(process, "pid", -1)) == pid:
                    used = int(getattr(process, "usedGpuMemory", 0))
                    current_used += used
                    peak_used = max(peak_used, used)
        if current_used <= 0 and peak_used <= 0:
            return {}
        return {
            f"{prefix}_bytes": current_used,
            f"{prefix}_peak_bytes": peak_used,
        }
    except Exception:
        return {}


def estimate_torch_model_bytes(model: Any) -> int | None:
    """Estimate parameter and buffer bytes for a PyTorch module."""

    total = 0
    found = False
    for iterator_name in ("parameters", "buffers"):
        iterator = getattr(model, iterator_name, None)
        if not callable(iterator):
            continue
        try:
            values = iterator()
        except Exception:
            continue
        for value in values:
            numel = getattr(value, "numel", None)
            element_size = getattr(value, "element_size", None)
            if callable(numel) and callable(element_size):
                total += int(numel()) * int(element_size())
                found = True
    return total if found else None


def estimate_npz_param_bytes(loaded_param_arrays: Any, param_names: list[str]) -> int:
    """Estimate parameter bytes from arrays in an np.load archive."""

    total = 0
    for index, _ in enumerate(param_names):
        array = loaded_param_arrays[f"arr_{index}"]
        total += int(getattr(array, "nbytes", 0))
    return total


def estimate_effective_bandwidth_gbps(bytes_per_step: int | None, steps: int, latency_s: float) -> float | None:
    """Estimate lower-bound effective bandwidth in GB/s."""

    if bytes_per_step is None or bytes_per_step <= 0 or steps <= 0 or latency_s <= 0:
        return None
    return float(bytes_per_step) * float(steps) / float(latency_s) / 1e9


def compact_optional_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Drop None values while preserving zero values."""

    return {key: value for key, value in metrics.items() if value is not None}
