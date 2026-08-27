"""Structured metrics for sparse history indexing and execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TimingBreakdown:
    index_s: float = 0.0
    q_summary_s: float = 0.0
    routing_s: float = 0.0
    cpu_gather_s: float = 0.0
    h2d_s: float = 0.0
    rope_s: float = 0.0
    attention_s: float = 0.0
    total_s: float = 0.0

    def add(self, other: "TimingBreakdown") -> None:
        for name in asdict(self):
            setattr(self, name, float(getattr(self, name)) + float(getattr(other, name)))


@dataclass
class SparseCallRecord:
    layer_id: int
    method: str
    candidate_frames: int
    candidate_history_tokens: int
    selected_history_tokens: int
    exact_tokens: int
    query_tokens: int
    dense_k_tokens: int
    executed_k_tokens: int
    transferred_bytes: int
    index_bytes: int
    candidate_transfer_bytes: int = 0
    full_history_pairs: int | None = None
    selected_history_pairs: int | None = None
    dense_qk_pairs_value: int | None = None
    executed_qk_pairs_value: int | None = None
    cluster_size_min: int | None = None
    cluster_size_max: int | None = None
    selected_units: int = 0
    candidate_units: int = 0
    staging_padding_tokens: int = 0
    attention_backend: str = "unknown"
    routing_stage: str = "unknown"
    history_pair_density_value: float | None = None
    history_transfer_density: float | None = None
    scheduled_pairs: int | None = None
    route_plan_sha256: str | None = None
    timing: TimingBreakdown = field(default_factory=TimingBreakdown)

    @property
    def history_density(self) -> float:
        if self.full_history_pairs is not None and self.selected_history_pairs is not None:
            if self.full_history_pairs == 0:
                return 1.0
            return self.selected_history_pairs / self.full_history_pairs
        if self.history_pair_density_value is not None:
            return self.history_pair_density_value
        if self.candidate_history_tokens == 0:
            return 1.0
        return self.selected_history_tokens / self.candidate_history_tokens

    @property
    def global_executed_density(self) -> float:
        denominator = (
            self.dense_qk_pairs_value
            if self.dense_qk_pairs_value is not None
            else self.query_tokens * self.dense_k_tokens
        )
        if denominator == 0:
            return 1.0
        numerator = (
            self.executed_qk_pairs_value
            if self.executed_qk_pairs_value is not None
            else self.query_tokens * self.executed_k_tokens
        )
        return numerator / denominator

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["history_density"] = self.history_density
        payload["history_pair_density"] = self.history_density
        payload["global_executed_density"] = self.global_executed_density
        if self.history_transfer_density is None:
            payload["history_transfer_density"] = (
                self.transferred_bytes / self.candidate_transfer_bytes
                if self.candidate_transfer_bytes
                else None
            )
        return payload


@dataclass
class SparseRunStats:
    """Accumulator whose density denominators are actual token/Q-K counts."""

    method: str
    attention_backend: str = "unknown"
    calls: int = 0
    indexed_frames: int = 0
    candidate_history_tokens: int = 0
    selected_history_tokens: int = 0
    full_history_pairs: int = 0
    selected_history_pairs: int = 0
    exact_tokens: int = 0
    dense_qk_pairs: int = 0
    executed_qk_pairs: int = 0
    archive_bytes: int = 0
    index_bytes: int = 0
    index_transfer_bytes: int = 0
    transferred_bytes: int = 0
    candidate_transfer_bytes: int = 0
    staging_padding_tokens: int = 0
    failed_calls: int = 0
    dense_fallback_calls: int = 0
    cluster_size_min: int | None = None
    cluster_size_max: int | None = None
    timing: TimingBreakdown = field(default_factory=TimingBreakdown)
    call_records: list[dict[str, Any]] = field(default_factory=list)
    routing_stage_counts: dict[str, int] = field(default_factory=dict)
    backend_counts: dict[str, int] = field(default_factory=dict)
    route_plan_sha256_counts: dict[str, int] = field(default_factory=dict)

    @property
    def history_density(self) -> float:
        if self.full_history_pairs == 0:
            return 1.0
        return self.selected_history_pairs / self.full_history_pairs

    @property
    def history_transfer_density(self) -> float | None:
        if self.candidate_transfer_bytes == 0:
            return None
        return self.transferred_bytes / self.candidate_transfer_bytes

    @property
    def global_executed_density(self) -> float:
        if self.dense_qk_pairs == 0:
            return 1.0
        return self.executed_qk_pairs / self.dense_qk_pairs

    def record_index(self, *, archive_bytes: int, index_bytes: int, elapsed_s: float) -> None:
        self.indexed_frames += 1
        self.archive_bytes += int(archive_bytes)
        self.index_bytes += int(index_bytes)
        self.timing.index_s += float(elapsed_s)

    def record_call(self, record: SparseCallRecord, *, keep_detail: bool = True) -> None:
        self.calls += 1
        self.candidate_history_tokens += int(record.candidate_history_tokens)
        self.selected_history_tokens += int(record.selected_history_tokens)
        self.full_history_pairs += int(
            record.full_history_pairs
            if record.full_history_pairs is not None
            else record.query_tokens * record.candidate_history_tokens
        )
        self.selected_history_pairs += int(
            record.selected_history_pairs
            if record.selected_history_pairs is not None
            else record.query_tokens * record.selected_history_tokens
        )
        self.exact_tokens += int(record.exact_tokens)
        self.dense_qk_pairs += int(
            record.dense_qk_pairs_value
            if record.dense_qk_pairs_value is not None
            else record.query_tokens * record.dense_k_tokens
        )
        self.executed_qk_pairs += int(
            record.executed_qk_pairs_value
            if record.executed_qk_pairs_value is not None
            else record.query_tokens * record.executed_k_tokens
        )
        self.transferred_bytes += int(record.transferred_bytes)
        self.candidate_transfer_bytes += int(record.candidate_transfer_bytes)
        self.index_transfer_bytes += int(record.index_bytes)
        self.staging_padding_tokens += int(record.staging_padding_tokens)
        self.attention_backend = record.attention_backend
        self.routing_stage_counts[record.routing_stage] = self.routing_stage_counts.get(record.routing_stage, 0) + 1
        self.backend_counts[record.attention_backend] = self.backend_counts.get(record.attention_backend, 0) + 1
        if record.route_plan_sha256:
            self.route_plan_sha256_counts[record.route_plan_sha256] = (
                self.route_plan_sha256_counts.get(record.route_plan_sha256, 0) + 1
            )
        self.timing.add(record.timing)
        if record.cluster_size_min is not None:
            self.cluster_size_min = (
                record.cluster_size_min
                if self.cluster_size_min is None
                else min(self.cluster_size_min, record.cluster_size_min)
            )
        if record.cluster_size_max is not None:
            self.cluster_size_max = (
                record.cluster_size_max
                if self.cluster_size_max is None
                else max(self.cluster_size_max, record.cluster_size_max)
            )
        if keep_detail:
            self.call_records.append(record.as_dict())

    def record_failure(self) -> None:
        self.failed_calls += 1

    def record_fallback(self) -> None:
        self.dense_fallback_calls += 1

    def merge(self, other: "SparseRunStats") -> None:
        if self.method != other.method:
            raise ValueError(f"cannot merge stats for {self.method} and {other.method}")
        for name in (
            "calls",
            "indexed_frames",
            "candidate_history_tokens",
            "selected_history_tokens",
            "full_history_pairs",
            "selected_history_pairs",
            "exact_tokens",
            "dense_qk_pairs",
            "executed_qk_pairs",
            "archive_bytes",
            "index_bytes",
            "index_transfer_bytes",
            "transferred_bytes",
            "candidate_transfer_bytes",
            "staging_padding_tokens",
            "failed_calls",
            "dense_fallback_calls",
        ):
            setattr(self, name, int(getattr(self, name)) + int(getattr(other, name)))
        self.timing.add(other.timing)
        self.attention_backend = other.attention_backend or self.attention_backend
        if other.cluster_size_min is not None:
            self.cluster_size_min = (
                other.cluster_size_min
                if self.cluster_size_min is None
                else min(self.cluster_size_min, other.cluster_size_min)
            )
        if other.cluster_size_max is not None:
            self.cluster_size_max = (
                other.cluster_size_max
                if self.cluster_size_max is None
                else max(self.cluster_size_max, other.cluster_size_max)
            )
        self.call_records.extend(other.call_records)
        for name, value in other.routing_stage_counts.items():
            self.routing_stage_counts[name] = self.routing_stage_counts.get(name, 0) + value
        for name, value in other.backend_counts.items():
            self.backend_counts[name] = self.backend_counts.get(name, 0) + value
        for name, value in other.route_plan_sha256_counts.items():
            self.route_plan_sha256_counts[name] = (
                self.route_plan_sha256_counts.get(name, 0) + value
            )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["history_density"] = self.history_density
        payload["history_pair_density"] = self.history_density
        payload["history_transfer_density"] = self.history_transfer_density
        payload["global_executed_density"] = self.global_executed_density
        return payload
