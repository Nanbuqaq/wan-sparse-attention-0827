"""Low-overhead profiling helpers and evidence-based bottleneck classification."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch


@contextlib.contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    pushed = False
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        pushed = True
    try:
        yield
    finally:
        if pushed:
            torch.cuda.nvtx.range_pop()


def process_rss_bytes() -> int | None:
    status = Path("/proc/self/status")
    if not status.is_file():
        return None
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


@dataclass
class PendingCudaInterval:
    name: str
    start: torch.cuda.Event
    end: torch.cuda.Event
    metadata: dict[str, Any] = field(default_factory=dict)


class DeferredCudaEventCollector:
    """Record CUDA service intervals and synchronize only once at flush."""

    def __init__(self, device: torch.device | str):
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("DeferredCudaEventCollector requires CUDA")
        self.pending: list[PendingCudaInterval] = []

    def begin(self) -> torch.cuda.Event:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def end(
        self,
        name: str,
        start: torch.cuda.Event,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.pending.append(
            PendingCudaInterval(name, start, end, dict(metadata or {}))
        )

    def flush(self) -> list[dict[str, Any]]:
        torch.cuda.synchronize(self.device)
        result = [
            {
                "name": interval.name,
                "service_s": interval.start.elapsed_time(interval.end) / 1000.0,
                "metadata": interval.metadata,
            }
            for interval in self.pending
        ]
        self.pending.clear()
        return result


def classify_bottleneck(
    exposed_seconds: dict[str, float],
    *,
    total_critical_path_s: float,
    dram_throughput_fraction: float | None = None,
    sm_throughput_fraction: float | None = None,
    dominant_threshold: float = 0.25,
    bubble_threshold: float = 0.15,
) -> dict[str, Any]:
    if total_critical_path_s <= 0:
        raise ValueError("total_critical_path_s must be positive")
    fractions = {
        name: max(0.0, float(value)) / total_critical_path_s
        for name, value in exposed_seconds.items()
    }
    labels = []
    if fractions.get("cpu_route_gather", 0.0) >= dominant_threshold:
        labels.append("cpu-bound")
    if fractions.get("host_device_transfer", 0.0) >= dominant_threshold:
        labels.append("transfer-bound")
    attention_fraction = fractions.get("attention", 0.0)
    if attention_fraction >= dominant_threshold:
        if dram_throughput_fraction is None or sm_throughput_fraction is None:
            labels.append("attention-bound-incomplete-counters")
        elif dram_throughput_fraction >= 0.70 and sm_throughput_fraction < 0.70:
            labels.append("hbm-bound")
        elif sm_throughput_fraction >= 0.70:
            labels.append("compute-bound")
        else:
            labels.append("attention-mixed")
    if fractions.get("pipeline_bubble", 0.0) >= bubble_threshold:
        labels.append("pipeline-bubble")
    if not labels:
        labels.append("mixed-bound")
    return {
        "labels": labels,
        "fractions": fractions,
        "total_critical_path_s": total_critical_path_s,
        "dram_throughput_fraction": dram_throughput_fraction,
        "sm_throughput_fraction": sm_throughput_fraction,
        "dominant_threshold": dominant_threshold,
        "bubble_threshold": bubble_threshold,
    }
