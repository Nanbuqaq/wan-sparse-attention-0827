"""Structured service-time and critical-path records for LongLive profiling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ServiceTimes:
    q_summary_s: float = 0.0
    query_d2h_s: float = 0.0
    archive_d2h_s: float = 0.0
    cpu_route_s: float = 0.0
    cpu_gather_pack_s: float = 0.0
    history_h2d_s: float = 0.0
    rope_s: float = 0.0
    attention_s: float = 0.0
    non_attention_s: float = 0.0
    vae_s: float = 0.0


@dataclass
class SystemTraceRecord:
    layer_id: int
    chunk_id: int
    denoising_pass: int
    route_plan_sha256: str | None
    transfer_plan_sha256: str | None
    execution_dataflow: str
    service: ServiceTimes = field(default_factory=ServiceTimes)
    predicted_exposed_wait_s: float | None = None
    measured_exposed_wait_s: float | None = None
    copy_compute_overlap_s: float = 0.0
    payload_bytes: int = 0
    padding_bytes: int = 0
    copy_count: int = 0
    cache_hit_bytes: int = 0
    cache_miss_bytes: int = 0
    host_rss_bytes: int | None = None
    gpu_allocated_bytes: int | None = None
    gpu_reserved_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "layer_id",
            "chunk_id",
            "denoising_pass",
            "payload_bytes",
            "padding_bytes",
            "copy_count",
            "cache_hit_bytes",
            "cache_miss_bytes",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.copy_compute_overlap_s < 0:
            raise ValueError("copy_compute_overlap_s must be non-negative")
        if self.measured_exposed_wait_s is not None and self.measured_exposed_wait_s < 0:
            raise ValueError("measured_exposed_wait_s must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
