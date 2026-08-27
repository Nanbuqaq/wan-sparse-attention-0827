"""Shared configuration, route-plan, and audit types."""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass, field
from statistics import median
from typing import Any

import torch


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


@dataclass(frozen=True)
class MethodConfig:
    method: str
    backend: str = "fixed64_bf16"
    density: float = 0.25
    parameter_origin: str = "exact_budget_1p3b_480p"
    q_clusters: int = 300
    k_clusters: int = 1000
    kmeans_init_iterations: int = 50
    kmeans_step_iterations: int = 2
    cluster_seed: int = 42
    block_size: int = 64
    top_p: float = 0.9
    min_k_ratio: float = 0.10
    official_first_timestep_fraction: float = 0.20
    official_first_layer_fraction: float = 0.03
    inference_steps: int = 50
    calls_per_step: int = 2
    measure_timing: bool = True
    route_params: dict[str, Any] = field(default_factory=dict)
    backend_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.density <= 1.0:
            raise ValueError("density must be in (0, 1]")
        if self.backend not in {
            "fixed64_bf16",
            "varlen_triton",
            "varlen_triton_native",
            "varlen_triton_csr",
        }:
            raise ValueError(f"unsupported backend: {self.backend}")
        if self.block_size != 64:
            raise ValueError("the reviewed fixed kernel requires block_size=64")
        if self.q_clusters <= 0 or self.k_clusters <= 0:
            raise ValueError("cluster counts must be positive")
        if self.kmeans_init_iterations <= 0 or self.kmeans_step_iterations <= 0:
            raise ValueError("k-means iteration counts must be positive")
        if self.calls_per_step <= 0:
            raise ValueError("calls_per_step must be positive")


@dataclass
class RoutePlan:
    method: str
    backend: str
    parameter_origin: str
    target_density: float
    block_map: torch.Tensor
    q_sizes: torch.Tensor
    k_sizes: torch.Tensor
    q_sorted_indices: torch.Tensor | None = None
    k_sorted_indices: torch.Tensor | None = None
    logical_pairs: int = 0
    total_pairs: int = 0
    scheduled_pairs: int = 0
    full_scheduled_pairs: int = 0
    padding_pairs: int = 0
    logical_density: float = 0.0
    scheduled_density_vs_dense: float = 0.0
    scheduled_fraction_of_full_tiles: float = 0.0
    padding_ratio: float = 0.0
    load_imbalance_cv: float = 0.0
    load_imbalance_max_mean: float = 0.0
    cluster_ms: float = 0.0
    permutation_ms: float = 0.0
    selection_ms: float = 0.0
    planner_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def graph_sha256(self) -> str:
        """Hash the concrete sparse graph and layout for cross-backend checks."""
        digest = hashlib.sha256()
        for tensor in (
            self.block_map.to(torch.uint8),
            self.q_sizes,
            self.k_sizes,
            self.q_sorted_indices,
            self.k_sorted_indices,
        ):
            if tensor is None:
                digest.update(b"<none>")
                continue
            value = tensor.detach().contiguous().cpu()
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "backend": self.backend,
            "parameter_origin": self.parameter_origin,
            "target_density": self.target_density,
            "logical_pairs": self.logical_pairs,
            "total_pairs": self.total_pairs,
            "logical_pair_density": self.logical_density,
            "scheduled_tile_pairs": self.scheduled_pairs,
            "full_scheduled_tile_pairs": self.full_scheduled_pairs,
            "scheduled_density_vs_dense": self.scheduled_density_vs_dense,
            "scheduled_fraction_of_full_tiles": self.scheduled_fraction_of_full_tiles,
            "padding_pairs": self.padding_pairs,
            "padding_ratio": self.padding_ratio,
            "load_imbalance_cv": self.load_imbalance_cv,
            "load_imbalance_max_mean": self.load_imbalance_max_mean,
            "q_block_count": int(self.q_sizes.shape[-1]),
            "k_block_count": int(self.k_sizes.shape[-1]),
            "q_size_min": int(self.q_sizes.min()),
            "q_size_max": int(self.q_sizes.max()),
            "k_size_min": int(self.k_sizes.min()),
            "k_size_max": int(self.k_sizes.max()),
            "cluster_ms": self.cluster_ms,
            "permutation_ms": self.permutation_ms,
            "selection_ms": self.selection_ms,
            "planner_ms": self.planner_ms,
            "metadata": self.metadata,
        }


@dataclass
class SparseRunStats:
    calls: int = 0
    sparse_kernel_calls: int = 0
    explicit_dense_reference_calls: int = 0
    failed_calls: int = 0
    dense_fallback_calls: int = 0
    logical_pairs: int = 0
    total_pairs: int = 0
    scheduled_pairs: int = 0
    full_scheduled_pairs: int = 0
    padding_pairs: int = 0
    cluster_times_ms: list[float] = field(default_factory=list)
    permutation_times_ms: list[float] = field(default_factory=list)
    selection_times_ms: list[float] = field(default_factory=list)
    kernel_times_ms: list[float] = field(default_factory=list)
    kernel_cold_times_ms: list[float] = field(default_factory=list)
    kernel_warm_times_ms: list[float] = field(default_factory=list)
    planner_times_ms: list[float] = field(default_factory=list)
    inverse_times_ms: list[float] = field(default_factory=list)
    attention_times_ms: list[float] = field(default_factory=list)
    density_errors: list[float] = field(default_factory=list)
    padding_ratios: list[float] = field(default_factory=list)
    load_imbalance_cvs: list[float] = field(default_factory=list)
    load_imbalance_max_means: list[float] = field(default_factory=list)
    method_counts: dict[str, int] = field(default_factory=dict)
    backend_counts: dict[str, int] = field(default_factory=dict)
    source_hashes: dict[str, str] = field(default_factory=dict)
    route_graph_hashes: dict[str, str] = field(default_factory=dict)
    _seen_kernel_signatures: set[str] = field(default_factory=set, repr=False)

    def record_plan(
        self,
        plan: RoutePlan,
        *,
        kernel_ms: float,
        inverse_ms: float,
        attention_ms: float,
    ) -> None:
        self.calls += 1
        self.sparse_kernel_calls += 1
        self.logical_pairs += plan.logical_pairs
        self.total_pairs += plan.total_pairs
        self.scheduled_pairs += plan.scheduled_pairs
        self.full_scheduled_pairs += plan.full_scheduled_pairs
        self.padding_pairs += plan.padding_pairs
        self.cluster_times_ms.append(plan.cluster_ms)
        self.permutation_times_ms.append(plan.permutation_ms)
        self.selection_times_ms.append(plan.selection_ms)
        self.planner_times_ms.append(plan.planner_ms)
        self.kernel_times_ms.append(kernel_ms)
        signature = (
            f"{plan.backend}:q{plan.q_sizes.shape[-1]}:k{plan.k_sizes.shape[-1]}:"
            f"qmax{int(plan.q_sizes.max())}:kmax{int(plan.k_sizes.max())}"
        )
        if signature in self._seen_kernel_signatures:
            self.kernel_warm_times_ms.append(kernel_ms)
        else:
            self._seen_kernel_signatures.add(signature)
            self.kernel_cold_times_ms.append(kernel_ms)
        self.inverse_times_ms.append(inverse_ms)
        self.attention_times_ms.append(attention_ms)
        self.density_errors.append(abs(plan.logical_density - plan.target_density))
        self.padding_ratios.append(plan.padding_ratio)
        self.load_imbalance_cvs.append(plan.load_imbalance_cv)
        self.load_imbalance_max_means.append(plan.load_imbalance_max_mean)
        self.method_counts[plan.method] = self.method_counts.get(plan.method, 0) + 1
        self.backend_counts[plan.backend] = self.backend_counts.get(plan.backend, 0) + 1

    def record_dense_reference(self, method: str, *, total_pairs: int) -> None:
        self.calls += 1
        self.explicit_dense_reference_calls += 1
        self.logical_pairs += total_pairs
        self.total_pairs += total_pairs
        self.scheduled_pairs += total_pairs
        self.full_scheduled_pairs += total_pairs
        self.method_counts[method] = self.method_counts.get(method, 0) + 1
        self.backend_counts["dense_reference"] = self.backend_counts.get("dense_reference", 0) + 1

    def _timing_summary(self, values: list[float]) -> dict[str, float | None]:
        return {
            "total_ms": float(sum(values)),
            "mean_ms": float(sum(values) / len(values)) if values else None,
            "p50_ms": float(median(values)) if values else None,
            "p90_ms": _percentile(values, 0.90),
        }

    def as_dict(self) -> dict[str, Any]:
        logical_density = self.logical_pairs / self.total_pairs if self.total_pairs else None
        scheduled_vs_dense = self.scheduled_pairs / self.total_pairs if self.total_pairs else None
        scheduled_fraction = (
            self.scheduled_pairs / self.full_scheduled_pairs if self.full_scheduled_pairs else None
        )
        padding_ratio = self.padding_pairs / self.scheduled_pairs if self.scheduled_pairs else None
        return {
            "calls": self.calls,
            "sparse_kernel_calls": self.sparse_kernel_calls,
            "explicit_dense_reference_calls": self.explicit_dense_reference_calls,
            "failed_calls": self.failed_calls,
            "dense_fallback_calls": self.dense_fallback_calls,
            "logical_pairs": self.logical_pairs,
            "total_pairs": self.total_pairs,
            "logical_pair_density": logical_density,
            "global_density": logical_density,
            "scheduled_tile_pairs": self.scheduled_pairs,
            "full_scheduled_tile_pairs": self.full_scheduled_pairs,
            "scheduled_density_vs_dense": scheduled_vs_dense,
            "scheduled_fraction_of_full_tiles": scheduled_fraction,
            "padding_pairs": self.padding_pairs,
            "padding_ratio": padding_ratio,
            "max_density_error": max(self.density_errors, default=0.0),
            "mean_padding_ratio_per_call": (
                sum(self.padding_ratios) / len(self.padding_ratios) if self.padding_ratios else None
            ),
            "mean_load_imbalance_cv": (
                sum(self.load_imbalance_cvs) / len(self.load_imbalance_cvs)
                if self.load_imbalance_cvs
                else None
            ),
            "max_load_imbalance_max_mean": max(self.load_imbalance_max_means, default=0.0),
            "timing": {
                "cluster": self._timing_summary(self.cluster_times_ms),
                "permutation": self._timing_summary(self.permutation_times_ms),
                "selection": self._timing_summary(self.selection_times_ms),
                "planner": self._timing_summary(self.planner_times_ms),
                "kernel": self._timing_summary(self.kernel_times_ms),
                "kernel_cold": self._timing_summary(self.kernel_cold_times_ms),
                "kernel_warm": self._timing_summary(self.kernel_warm_times_ms),
                "inverse": self._timing_summary(self.inverse_times_ms),
                "attention_total": self._timing_summary(self.attention_times_ms),
            },
            "method_counts": self.method_counts,
            "backend_counts": self.backend_counts,
            "source_hashes": self.source_hashes,
            "route_graph_hashes": self.route_graph_hashes,
        }
