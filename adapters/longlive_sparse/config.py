"""Configuration contract for sparse LongLive history attention."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .methods import BACKENDS, METHOD_SPECS, method_spec

_REFRESH_POLICIES = {"per_step", "per_chunk"}
_ROPE_POLICIES = {"upstream_zero", "recency_rank", "clipped_relative_age"}


@dataclass(frozen=True)
class SparseHistoryConfig:
    """All decisions that affect sparse history selection or execution.

    ``history_density`` is applied only to the coarse-retrieved historical
    candidate set.  Sink, current chunk, and the configured recent window are
    always exact and are accounted for separately in global executed density.
    """

    method: str = "block64_history"
    backend: str = "grouped_fa2"
    history_density: float = 0.25
    recent_exact_frames: int = 3
    block_size: int = 64
    clusters_per_frame: int = 32
    kmeans_iterations: int = 5
    kmeans_tolerance: float = 1e-4
    seed: int = 42
    refresh_policy: str = "per_step"
    rope_policy: str = "upstream_zero"
    max_relative_age: int = 31
    pin_memory: bool = True
    non_blocking_h2d: bool = True
    fail_on_fallback: bool = True
    record_per_call: bool = True

    def __post_init__(self) -> None:
        if self.method not in METHOD_SPECS:
            raise ValueError(f"unsupported sparse history method: {self.method!r}")
        if self.backend not in BACKENDS:
            raise ValueError(f"unsupported sparse history backend: {self.backend!r}")
        if not 0.0 < self.history_density <= 1.0:
            raise ValueError("history_density must be in (0, 1]")
        if self.recent_exact_frames < 0:
            raise ValueError("recent_exact_frames must be non-negative")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        if self.clusters_per_frame < 1:
            raise ValueError("clusters_per_frame must be positive")
        if self.kmeans_iterations < 1:
            raise ValueError("kmeans_iterations must be positive")
        if self.kmeans_tolerance < 0:
            raise ValueError("kmeans_tolerance must be non-negative")
        if self.refresh_policy not in _REFRESH_POLICIES:
            raise ValueError(f"unsupported refresh_policy: {self.refresh_policy!r}")
        if self.rope_policy not in _ROPE_POLICIES:
            raise ValueError(f"unsupported rope_policy: {self.rope_policy!r}")
        if self.max_relative_age < 0:
            raise ValueError("max_relative_age must be non-negative")

    @property
    def is_dense(self) -> bool:
        return self.method in {"native_dense", "rag_dense", "dense_history"} or self.history_density == 1.0

    @property
    def routing_stage(self) -> str:
        return method_spec(self.method).routing_stage

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "SparseHistoryConfig":
        return cls(**dict(value or {}))
