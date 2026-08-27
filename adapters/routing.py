"""Deterministic routing and exact executed-pair budgeting."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

import torch
import torch.nn.functional as F

from .types import MethodConfig, RoutePlan
from .vendor import load_svoo_core, load_svoo_permutation


T = TypeVar("T")
_SVOO_CORE = None
_SVOO_PERMUTE = None


def _core():
    global _SVOO_CORE
    if _SVOO_CORE is None:
        _SVOO_CORE = load_svoo_core()
    return _SVOO_CORE


def _permute_module():
    global _SVOO_PERMUTE
    if _SVOO_PERMUTE is None:
        _SVOO_PERMUTE = load_svoo_permutation()
    return _SVOO_PERMUTE


def _timed_cuda(fn: Callable[[], T], *, enabled: bool = True) -> tuple[T, float]:
    if not enabled or not torch.cuda.is_available():
        start = time.perf_counter()
        result = fn()
        return result, (time.perf_counter() - start) * 1000.0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    result = fn()
    end_event.record()
    torch.cuda.synchronize()
    return result, float(start_event.elapsed_time(end_event))


@dataclass
class RoutingState:
    q_centroids: torch.Tensor | None = None
    k_centroids: torch.Tensor | None = None
    extras: dict | None = None

    def cache(self) -> dict:
        if self.extras is None:
            self.extras = {}
        return self.extras


@dataclass
class KMeansResult:
    labels: torch.Tensor
    centroids: torch.Tensor
    sizes: torch.Tensor
    empty_clusters: int


def _deterministic_initial_centroids(
    flat: torch.Tensor,
    clusters: int,
    seed: int,
) -> torch.Tensor:
    if clusters > flat.shape[1]:
        raise ValueError(f"clusters={clusters} exceeds token count={flat.shape[1]}")
    generator = torch.Generator(device=flat.device)
    generator.manual_seed(seed)
    indices = torch.randint(
        0,
        flat.shape[1],
        (flat.shape[0], clusters),
        generator=generator,
        device=flat.device,
    )
    return torch.gather(
        flat,
        1,
        indices.unsqueeze(-1).expand(-1, -1, flat.shape[-1]),
    ).contiguous()


def batched_euclidean_kmeans(
    vectors: torch.Tensor,
    *,
    clusters: int,
    iterations: int,
    seed: int,
    initial_centroids: torch.Tensor | None = None,
) -> KMeansResult:
    """4090-safe SVG2-style Euclidean K-means over `[B,H,L,D]`."""
    if vectors.ndim != 4 or not vectors.is_cuda:
        raise ValueError("k-means expects CUDA [B,H,L,D]")
    batch, heads, length, dim = vectors.shape
    flat = vectors.float().reshape(batch * heads, length, dim).contiguous()
    centroids = (
        _deterministic_initial_centroids(flat, clusters, seed)
        if initial_centroids is None
        else initial_centroids.reshape(batch * heads, clusters, dim).float().contiguous()
    )
    x_sq = (flat * flat).sum(dim=-1)
    labels = None
    sizes = None
    for _ in range(iterations):
        labels = _core().euclid_assign_triton(
            flat,
            centroids,
            x_sq,
            BLOCK_N=64,
            BLOCK_K=64,
        )
        centroids, sizes = _core().triton_centroid_update_sorted_euclid(
            flat,
            labels,
            centroids,
            BLOCK_N=128,
        )
    if labels is None or sizes is None:
        raise RuntimeError("k-means produced no iteration")
    return KMeansResult(
        labels=labels.reshape(batch, heads, length).long(),
        centroids=centroids.reshape(batch, heads, clusters, dim),
        sizes=sizes.reshape(batch, heads, clusters).to(torch.int32),
        empty_clusters=int((sizes == 0).sum()),
    )


def _permute_by_labels(
    tensor: torch.Tensor,
    labels: torch.Tensor | None,
    *,
    sorted_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    flattened_labels = labels.flatten(0, 1) if labels is not None else None
    return _permute_module().permute_tensor_by_labels_triton(
        tensor,
        flattened_labels,
        dim=2,
        sorted_indices=sorted_indices,
    )


def inverse_permute(tensor: torch.Tensor, sorted_indices: torch.Tensor) -> torch.Tensor:
    return _permute_module().apply_inverse_permutation_triton(tensor, sorted_indices, dim=2)


def fixed_block_sizes(
    batch: int,
    heads: int,
    length: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    blocks = math.ceil(length / block_size)
    sizes = torch.full(
        (batch, heads, blocks),
        block_size,
        dtype=torch.int32,
        device=device,
    )
    if length % block_size:
        sizes[..., -1] = length % block_size
    return sizes


def _pad_to_blocks(tensor: torch.Tensor, block_size: int) -> torch.Tensor:
    padded = math.ceil(tensor.shape[2] / block_size) * block_size
    return F.pad(tensor, (0, 0, 0, padded - tensor.shape[2]))


def block_means(tensor: torch.Tensor, sizes: torch.Tensor, block_size: int) -> torch.Tensor:
    padded = _pad_to_blocks(tensor.float(), block_size)
    batch, heads, _, dim = padded.shape
    blocks = sizes.shape[-1]
    return padded.view(batch, heads, blocks, block_size, dim).sum(dim=3) / sizes.unsqueeze(-1).float()


def exact_pair_budget_map(
    scores: torch.Tensor,
    q_sizes: torch.Tensor,
    k_sizes: torch.Tensor,
    density: float,
) -> torch.Tensor:
    """Choose pairs per query block, then repair the global actual-pair budget.

    SVG2's Top-p decision is made independently for every query cluster.  A purely
    global ranking can starve low-norm query clusters even when its pair count is
    exact, so the initial allocation targets the same K-token budget per query
    block. The closest cumulative K-token boundary is selected independently
    for every row; this is fully GPU-vectorized and avoids Python scalar loops
    in the 3,000-call Wan path.
    """
    if scores.ndim != 4:
        raise ValueError("scores must be [B,H,Q,K]")
    batch, heads, q_blocks, k_blocks = scores.shape
    result = torch.zeros_like(scores, dtype=torch.bool)
    for b in range(batch):
        for h in range(heads):
            qh = q_sizes[b, h].long()
            kh = k_sizes[b, h].long()
            score = scores[b, h].float().clone()
            score[:, kh == 0] = -float("inf")
            order = torch.argsort(score, dim=1, descending=True, stable=True)
            ordered_k_sizes = kh.index_select(0, order.flatten()).view(q_blocks, k_blocks)
            cumulative_k = torch.cumsum(ordered_k_sizes, dim=1)
            row_target = int(round(int(kh.sum()) * density))
            row_counts = (cumulative_k - row_target).abs().argmin(dim=1) + 1
            row_counts = torch.where(qh > 0, row_counts, torch.zeros_like(row_counts))
            positions = torch.arange(k_blocks, device=scores.device).view(1, k_blocks)
            selected_sorted = positions < row_counts.unsqueeze(1)
            result[b, h].scatter_(1, order, selected_sorted)
    return result


def global_exact_pair_budget_map(
    scores: torch.Tensor,
    q_sizes: torch.Tensor,
    k_sizes: torch.Tensor,
    density: float,
) -> torch.Tensor:
    """Global exact-area selector with one mandatory key block per query row."""
    batch, heads, _, _ = scores.shape
    result = torch.zeros_like(scores, dtype=torch.bool)
    for b in range(batch):
        for h in range(heads):
            qh = q_sizes[b, h].long()
            kh = k_sizes[b, h].long()
            score = scores[b, h].float().clone()
            score[:, kh == 0] = -float("inf")
            areas = qh[:, None] * kh[None, :]
            target = int(round(int(qh.sum()) * int(kh.sum()) * density))
            rows = torch.nonzero(qh > 0, as_tuple=False).flatten()
            best = score.argmax(dim=1)
            result[b, h, rows, best.index_select(0, rows)] = True
            mandatory = int((areas * result[b, h]).sum())
            remaining = score.masked_fill(result[b, h], -float("inf")).flatten()
            order = torch.argsort(remaining, descending=True, stable=True)
            cumulative = mandatory + torch.cumsum(areas.flatten().index_select(0, order), dim=0)
            errors = torch.cat(
                (
                    torch.tensor([abs(target - mandatory)], device=scores.device, dtype=torch.long),
                    (cumulative - target).abs(),
                )
            )
            count = int(errors.argmin())
            if count:
                result[b, h].view(-1)[order[:count]] = True
    return result


def fixed_edge_budget_map(scores: torch.Tensor, density: float) -> torch.Tensor:
    """Fast exact block-edge budget with an even per-query-row allocation."""
    batch, heads, q_blocks, k_blocks = scores.shape
    target_edges = max(1, min(q_blocks * k_blocks, int(round(q_blocks * k_blocks * density))))
    base = target_edges // q_blocks
    extra = target_edges - base * q_blocks
    result = torch.zeros_like(scores, dtype=torch.bool)
    if base:
        indices = torch.topk(scores, k=base, dim=-1).indices
        result.scatter_(-1, indices, True)
    if extra:
        remaining = scores.masked_fill(result, -float("inf"))
        values, indices = remaining.max(dim=-1)
        rows = torch.topk(values, k=extra, dim=-1).indices
        batch_ids = torch.arange(batch, device=scores.device).view(batch, 1, 1)
        head_ids = torch.arange(heads, device=scores.device).view(1, heads, 1)
        result[batch_ids, head_ids, rows, indices.gather(-1, rows)] = True
    return result


def top_p_map(
    scores: torch.Tensor,
    k_sizes: torch.Tensor,
    *,
    top_p: float,
    min_k_ratio: float,
) -> torch.Tensor:
    weighted_logits = scores.float() + torch.log(k_sizes.unsqueeze(-2).float().clamp_min(1.0))
    weighted_logits = weighted_logits.masked_fill(k_sizes.unsqueeze(-2) == 0, -float("inf"))
    probabilities = torch.softmax(weighted_logits, dim=-1)
    sorted_probs, order = torch.sort(probabilities, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep = cumulative <= top_p
    keep[..., 1:] = keep[..., :-1].clone()
    keep[..., 0] = True
    minimum = max(1, int(round(k_sizes.shape[-1] * min_k_ratio)))
    keep[..., :minimum] = True
    result = torch.zeros_like(keep)
    result.scatter_(-1, order, keep)
    return result


def calibrated_top_p_map(
    scores: torch.Tensor,
    q_sizes: torch.Tensor,
    k_sizes: torch.Tensor,
    *,
    target_density: float,
    min_k_ratio: float = 0.0,
    search_steps: int = 16,
) -> tuple[torch.Tensor, float]:
    """Calibrate one Top-p threshold to the actual cluster-pair budget."""
    weighted_logits = scores.float() + torch.log(k_sizes.unsqueeze(-2).float().clamp_min(1.0))
    weighted_logits = weighted_logits.masked_fill(k_sizes.unsqueeze(-2) == 0, -float("inf"))
    probabilities = torch.softmax(weighted_logits, dim=-1)
    sorted_probs, order = torch.sort(probabilities, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    areas = q_sizes.long().unsqueeze(-1) * k_sizes.long().unsqueeze(-2)
    sorted_areas = torch.gather(areas, -1, order)
    total = int(areas.sum())
    target = int(round(total * target_density))
    low, high = 0.0, 1.0
    best_keep = None
    best_p = 0.0
    best_error = None
    minimum = max(1, int(round(k_sizes.shape[-1] * min_k_ratio)))
    for _ in range(search_steps):
        threshold = (low + high) / 2.0
        keep = cumulative <= threshold
        keep[..., 1:] = keep[..., :-1].clone()
        keep[..., 0] = True
        keep[..., :minimum] = True
        selected = int((sorted_areas * keep).sum())
        error = abs(selected - target)
        if best_error is None or error < best_error:
            best_error = error
            best_keep = keep.clone()
            best_p = threshold
        if selected < target:
            low = threshold
        else:
            high = threshold
    if best_keep is None:
        raise RuntimeError("Top-p calibration failed")
    # Preserve the mandatory top `minimum` clusters per query row, then repair
    # the remaining global area using the lowest/highest-probability boundary
    # pairs. This keeps the adaptive Top-p allocation while meeting the actual
    # Q-K token-pair budget much more closely than a p-only binary search.
    current = int((sorted_areas * best_keep).sum())
    flat_keep = best_keep.view(-1)
    flat_areas = sorted_areas.view(-1)
    flat_priority = sorted_probs.view(-1)
    positions = torch.arange(k_sizes.shape[-1], device=scores.device).view(
        *([1] * (best_keep.ndim - 1)), k_sizes.shape[-1]
    )
    protected = positions < minimum
    protected = protected.expand_as(best_keep).reshape(-1)
    if current > target:
        candidates = torch.nonzero(flat_keep & ~protected & (flat_areas > 0), as_tuple=False).flatten()
        if candidates.numel():
            remove_order = candidates.index_select(
                0, torch.argsort(flat_priority.index_select(0, candidates), descending=False, stable=True)
            )
            cumulative = torch.cumsum(flat_areas.index_select(0, remove_order), dim=0)
            errors = torch.cat(
                (
                    torch.tensor([abs(current - target)], device=scores.device, dtype=torch.long),
                    (current - cumulative - target).abs(),
                )
            )
            count = int(errors.argmin())
            if count:
                flat_keep[remove_order[:count]] = False
    elif current < target:
        candidates = torch.nonzero(~flat_keep & (flat_areas > 0), as_tuple=False).flatten()
        if candidates.numel():
            add_order = candidates.index_select(
                0, torch.argsort(flat_priority.index_select(0, candidates), descending=True, stable=True)
            )
            cumulative = torch.cumsum(flat_areas.index_select(0, add_order), dim=0)
            errors = torch.cat(
                (
                    torch.tensor([abs(target - current)], device=scores.device, dtype=torch.long),
                    (current + cumulative - target).abs(),
                )
            )
            count = int(errors.argmin())
            if count:
                flat_keep[add_order[:count]] = True
    result = torch.zeros_like(best_keep)
    result.scatter_(-1, order, best_keep)
    return result, best_p


def _plan_metrics(
    *,
    method: str,
    backend: str,
    parameter_origin: str,
    density: float,
    block_map: torch.Tensor,
    q_sizes: torch.Tensor,
    k_sizes: torch.Tensor,
    q_sorted_indices: torch.Tensor | None,
    k_sorted_indices: torch.Tensor | None,
    cluster_ms: float,
    permutation_ms: float,
    selection_ms: float,
    metadata: dict,
) -> RoutePlan:
    areas = q_sizes.long().unsqueeze(-1) * k_sizes.long().unsqueeze(-2)
    logical_pairs = int((areas * block_map).sum())
    total_pairs = int(areas.sum())
    if backend == "fixed64_bf16":
        q_tile = torch.full_like(q_sizes, 64)
        k_tile = torch.full_like(k_sizes, 64)
    elif backend in {"varlen_triton", "varlen_triton_native", "varlen_triton_csr"}:
        if backend == "varlen_triton_csr":
            backend_params = metadata.get("backend_params", {})
            block_m = int(backend_params.get("block_m", 64))
            block_n = int(backend_params.get("block_n", 32))
        else:
            block_m, block_n = 128, 64
        q_tile = torch.div(q_sizes + block_m - 1, block_m, rounding_mode="floor") * block_m
        k_tile = torch.div(k_sizes + block_n - 1, block_n, rounding_mode="floor") * block_n
    else:
        raise ValueError(backend)
    scheduled_areas = q_tile.long().unsqueeze(-1) * k_tile.long().unsqueeze(-2)
    scheduled_pairs = int((scheduled_areas * block_map).sum())
    full_scheduled_pairs = int(scheduled_areas.sum())
    padding_pairs = max(0, scheduled_pairs - logical_pairs)
    loads = q_tile.float() * (
        block_map.float() * k_tile.float().unsqueeze(-2)
    ).sum(dim=-1)
    nonzero = loads[loads > 0]
    if nonzero.numel():
        mean = nonzero.mean()
        cv = float(nonzero.std(unbiased=False) / mean) if nonzero.numel() > 1 else 0.0
        max_mean = float(nonzero.max() / mean)
    else:
        cv = 0.0
        max_mean = 0.0
    return RoutePlan(
        method=method,
        backend=backend,
        parameter_origin=parameter_origin,
        target_density=density,
        block_map=block_map.contiguous(),
        q_sizes=q_sizes.contiguous(),
        k_sizes=k_sizes.contiguous(),
        q_sorted_indices=q_sorted_indices,
        k_sorted_indices=k_sorted_indices,
        logical_pairs=logical_pairs,
        total_pairs=total_pairs,
        scheduled_pairs=scheduled_pairs,
        full_scheduled_pairs=full_scheduled_pairs,
        padding_pairs=padding_pairs,
        logical_density=logical_pairs / total_pairs,
        scheduled_density_vs_dense=scheduled_pairs / total_pairs,
        scheduled_fraction_of_full_tiles=scheduled_pairs / full_scheduled_pairs,
        padding_ratio=padding_pairs / scheduled_pairs if scheduled_pairs else 0.0,
        load_imbalance_cv=cv,
        load_imbalance_max_mean=max_mean,
        cluster_ms=cluster_ms,
        permutation_ms=permutation_ms,
        selection_ms=selection_ms,
        metadata=metadata,
    )


def _cluster_route_plan(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    config: MethodConfig,
    method: str,
    q_result: KMeansResult,
    k_result: KMeansResult,
    cluster_scores: torch.Tensor,
    cluster_ms: float,
    metadata: dict,
):
    metadata = dict(metadata)
    metadata["backend_params"] = dict(config.backend_params)
    selected_map, selection_ms = _timed_cuda(
        lambda: exact_pair_budget_map(
            cluster_scores,
            q_result.sizes,
            k_result.sizes,
            config.density,
        ),
        enabled=config.measure_timing,
    )
    if config.backend == "fixed64_bf16" or config.route_params.get("materialization") == "fixed64_graph":
        return _fixed_plan(
            query,
            key,
            value,
            config=config,
            method=method,
            q_labels=q_result.labels,
            k_labels=k_result.labels,
            cluster_priority=cluster_scores,
            cluster_adaptive_map=selected_map,
            cluster_ms=cluster_ms,
            preselection_ms=selection_ms,
            metadata=metadata,
        )

    def permute():
        q_work, q_order = _permute_by_labels(query, q_result.labels)
        k_work, k_order = _permute_by_labels(key, k_result.labels)
        v_work, _ = _permute_by_labels(value, None, sorted_indices=k_order)
        return q_work, k_work, v_work, q_order, k_order

    (q_work, k_work, v_work, q_order, k_order), permutation_ms = _timed_cuda(
        permute, enabled=config.measure_timing
    )
    plan = _plan_metrics(
        method=method,
        backend=config.backend,
        parameter_origin=config.parameter_origin,
        density=config.density,
        block_map=selected_map,
        q_sizes=q_result.sizes,
        k_sizes=k_result.sizes,
        q_sorted_indices=q_order,
        k_sorted_indices=k_order,
        cluster_ms=cluster_ms,
        permutation_ms=permutation_ms,
        selection_ms=selection_ms,
        metadata=metadata,
    )
    plan.metadata["original_length"] = query.shape[2]
    return q_work, k_work, v_work, plan


def _fixed_plan(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    config: MethodConfig,
    method: str,
    q_labels: torch.Tensor | None = None,
    k_labels: torch.Tensor | None = None,
    cluster_priority: torch.Tensor | None = None,
    cluster_adaptive_map: torch.Tensor | None = None,
    cluster_ms: float = 0.0,
    preselection_ms: float = 0.0,
    metadata: dict | None = None,
    score_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RoutePlan]:
    batch, heads, length, _ = query.shape

    def do_permutation():
        q_work, q_order = (
            _permute_by_labels(query, q_labels) if q_labels is not None else (query, None)
        )
        if k_labels is not None:
            k_work, k_order = _permute_by_labels(key, k_labels)
            v_work, _ = _permute_by_labels(value, None, sorted_indices=k_order)
        else:
            k_work, v_work, k_order = key, value, None
        return q_work, k_work, v_work, q_order, k_order

    (q_work, k_work, v_work, q_order, k_order), permutation_ms = _timed_cuda(
        do_permutation, enabled=config.measure_timing
    )
    sizes = fixed_block_sizes(batch, heads, length, config.block_size, query.device)

    def select():
        q_means = block_means(q_work, sizes, config.block_size)
        k_means = block_means(k_work, sizes, config.block_size)
        direct_scores = torch.matmul(q_means.float(), k_means.float().transpose(-2, -1)) / math.sqrt(query.shape[-1])
        if score_override is not None:
            if score_override.shape != direct_scores.shape:
                raise ValueError("fixed score override shape mismatch")
            direct_scores = score_override.to(direct_scores)
        if cluster_priority is None or cluster_adaptive_map is None:
            return fixed_edge_budget_map(direct_scores, config.density)
        if q_order is None or k_order is None or q_labels is None or k_labels is None:
            raise RuntimeError("cluster-priority fixed routing requires Q/K permutations")
        sorted_q_labels = torch.gather(
            q_labels.flatten(0, 1), 1, q_order.long()
        ).reshape(batch, heads, length)
        sorted_k_labels = torch.gather(
            k_labels.flatten(0, 1), 1, k_order.long()
        ).reshape(batch, heads, length)
        centers = (
            torch.arange(sizes.shape[-1], device=query.device) * config.block_size
            + sizes[0, 0].long() // 2
        ).clamp_max(length - 1)
        q_block_labels = sorted_q_labels.index_select(2, centers)
        k_block_labels = sorted_k_labels.index_select(2, centers)
        priority = torch.empty_like(direct_scores)
        selected = torch.empty_like(direct_scores, dtype=torch.bool)
        for b in range(batch):
            for h in range(heads):
                q_index = q_block_labels[b, h].view(-1, 1)
                k_index = k_block_labels[b, h].view(1, -1)
                priority[b, h] = cluster_priority[b, h][q_index, k_index]
                selected[b, h] = cluster_adaptive_map[b, h][q_index, k_index]
        combined = priority + selected.float() * 1e4 + direct_scores * 1e-4
        return global_exact_pair_budget_map(combined, sizes, sizes, config.density)

    block_map, selection_ms = _timed_cuda(select, enabled=config.measure_timing)
    if config.backend == "fixed64_bf16":
        q_exec = _pad_to_blocks(q_work, config.block_size)
        k_exec = _pad_to_blocks(k_work, config.block_size)
        v_exec = _pad_to_blocks(v_work, config.block_size)
    else:
        q_exec, k_exec, v_exec = q_work, k_work, v_work
    plan = _plan_metrics(
        method=method,
        backend=config.backend,
        parameter_origin=config.parameter_origin,
        density=config.density,
        block_map=block_map,
        q_sizes=sizes,
        k_sizes=sizes,
        q_sorted_indices=q_order,
        k_sorted_indices=k_order,
        cluster_ms=cluster_ms,
        permutation_ms=permutation_ms,
        selection_ms=selection_ms + preselection_ms,
        metadata=metadata or {},
    )
    plan.metadata["original_length"] = length
    return q_exec, k_exec, v_exec, plan


def _oracle_fixed_plan(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    config: MethodConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RoutePlan]:
    batch, heads, length, dim = query.shape
    sizes = fixed_block_sizes(batch, heads, length, config.block_size, query.device)
    q_means = block_means(query, sizes, config.block_size)
    k_means = block_means(key, sizes, config.block_size)
    blocks = sizes.shape[-1]

    def select():
        route_scores = torch.empty((batch, heads, blocks, blocks), device=query.device, dtype=torch.float32)
        budget = max(1, min(length, int(round(length * config.density))))
        for b in range(batch):
            for h in range(heads):
                token_scores = q_means[b, h].float() @ key[b, h].float().T / math.sqrt(dim)
                selected = torch.topk(token_scores, k=budget, dim=1).indices
                selected_blocks = torch.div(selected, config.block_size, rounding_mode="floor")
                overlap = torch.zeros((blocks, blocks), device=query.device, dtype=torch.float32)
                overlap.scatter_add_(
                    1,
                    selected_blocks,
                    torch.ones_like(selected_blocks, dtype=torch.float32),
                )
                direct = q_means[b, h].float() @ k_means[b, h].float().T / math.sqrt(dim)
                route_scores[b, h] = overlap + direct * 1e-4
        return fixed_edge_budget_map(route_scores, config.density)

    block_map, selection_ms = _timed_cuda(select, enabled=config.measure_timing)
    plan = _plan_metrics(
        method="token_oracle",
        backend="fixed64_bf16",
        parameter_origin=config.parameter_origin,
        density=config.density,
        block_map=block_map,
        q_sizes=sizes,
        k_sizes=sizes,
        q_sorted_indices=None,
        k_sorted_indices=None,
        cluster_ms=0.0,
        permutation_ms=0.0,
        selection_ms=selection_ms,
        metadata={
            "deployable": False,
            "selector": "dense_qblock_to_all_tokens_then_fixed64",
            "token_budget_per_query_block": max(1, int(round(length * config.density))),
        },
    )
    plan.metadata["original_length"] = length
    return (
        _pad_to_blocks(query, config.block_size),
        _pad_to_blocks(key, config.block_size),
        _pad_to_blocks(value, config.block_size),
        plan,
    )


def _route_attention_legacy(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    config: MethodConfig,
    state: RoutingState,
    layer: int,
    call_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RoutePlan]:
    if query.shape != key.shape or query.shape != value.shape or query.ndim != 4:
        raise ValueError("routing expects matching [B,H,L,D] Q/K/V")
    if query.dtype is not torch.bfloat16 or key.dtype is not torch.bfloat16 or value.dtype is not torch.bfloat16:
        raise TypeError("sparse routing requires BF16 Q/K/V")
    method = config.method
    if method == "original_block":
        return _fixed_plan(query, key, value, config=config, method=method)
    if method == "token_oracle":
        return _oracle_fixed_plan(query, key, value, config)

    fixed_k_clusters = None
    if method == "fixed_k128":
        fixed_k_clusters = 128
    elif method == "fixed_k256":
        fixed_k_clusters = 256

    first_call = state.k_centroids is None or (fixed_k_clusters is None and state.q_centroids is None)
    iterations = config.kmeans_init_iterations if first_call else config.kmeans_step_iterations

    if fixed_k_clusters is not None:
        def cluster_k():
            return batched_euclidean_kmeans(
                key,
                clusters=fixed_k_clusters,
                iterations=iterations,
                seed=config.cluster_seed + layer * 1009,
                initial_centroids=state.k_centroids,
            )

        k_result, cluster_ms = _timed_cuda(cluster_k, enabled=config.measure_timing)
        state.k_centroids = k_result.centroids.detach()
        return _fixed_plan(
            query,
            key,
            value,
            config=config,
            method=method,
            k_labels=k_result.labels,
            cluster_ms=cluster_ms,
            metadata={
                "k_clusters": fixed_k_clusters,
                "iterations": iterations,
                "empty_k_clusters": k_result.empty_clusters,
            },
        )

    if method not in {"svg2", "svg2_fixed", "svg2_varlen", "svg2_official_top_p"}:
        raise ValueError(f"unknown sparse method: {method}")

    def cluster_qk():
        q_result = batched_euclidean_kmeans(
            query,
            clusters=config.q_clusters,
            iterations=iterations,
            seed=config.cluster_seed + layer * 1009 + call_index * 17,
            initial_centroids=state.q_centroids,
        )
        k_result = batched_euclidean_kmeans(
            key,
            clusters=config.k_clusters,
            iterations=iterations,
            seed=config.cluster_seed + 97 + layer * 1009 + call_index * 17,
            initial_centroids=state.k_centroids,
        )
        return q_result, k_result

    (q_result, k_result), cluster_ms = _timed_cuda(cluster_qk, enabled=config.measure_timing)
    state.q_centroids = q_result.centroids.detach()
    state.k_centroids = k_result.centroids.detach()
    metadata = {
        "q_clusters": config.q_clusters,
        "k_clusters": config.k_clusters,
        "iterations": iterations,
        "empty_q_clusters": q_result.empty_clusters,
        "empty_k_clusters": k_result.empty_clusters,
        "backend_params": dict(config.backend_params),
    }
    centroid_scores = torch.matmul(
        q_result.centroids.float(), k_result.centroids.float().transpose(-2, -1)
    ) / math.sqrt(query.shape[-1])
    cluster_priority = centroid_scores + torch.log(
        k_result.sizes.unsqueeze(-2).float().clamp_min(1.0)
    )
    calibrated_map = None
    calibrated_p = None
    calibration_ms = 0.0
    if method in {"svg2", "svg2_fixed", "svg2_varlen"}:
        # At 5%-15%, a count-based cluster floor can itself exceed the token
        # budget because cluster sizes are imbalanced. Keep min-K only for the
        # 20%-25% quality region; low-density comparisons prioritize exact
        # executed pairs.
        effective_min_k_ratio = config.min_k_ratio if config.density >= 0.20 else 0.0
        (calibrated_map, calibrated_p), calibration_ms = _timed_cuda(
            lambda: calibrated_top_p_map(
                centroid_scores,
                q_result.sizes,
                k_result.sizes,
                target_density=config.density,
                min_k_ratio=effective_min_k_ratio,
            ),
            enabled=config.measure_timing,
        )
        metadata["calibrated_top_p"] = calibrated_p
        metadata["calibrated_min_k_ratio"] = effective_min_k_ratio
    if method == "svg2" and (
        config.backend == "fixed64_bf16"
        or config.route_params.get("materialization") == "fixed64_graph"
    ):
        return _fixed_plan(
            query,
            key,
            value,
            config=config,
            method="svg2",
            q_labels=q_result.labels,
            k_labels=k_result.labels,
            cluster_priority=cluster_priority,
            cluster_adaptive_map=calibrated_map,
            cluster_ms=cluster_ms,
            preselection_ms=calibration_ms,
            metadata=metadata,
        )
    if method == "svg2":
        method = "svg2_varlen"
    if method == "svg2_fixed":
        return _fixed_plan(
            query,
            key,
            value,
            config=config,
            method=method,
            q_labels=q_result.labels,
            k_labels=k_result.labels,
            cluster_priority=cluster_priority,
            cluster_adaptive_map=calibrated_map,
            cluster_ms=cluster_ms,
            preselection_ms=calibration_ms,
            metadata=metadata,
        )

    def do_permutation():
        q_work, q_order = _permute_by_labels(query, q_result.labels)
        k_work, k_order = _permute_by_labels(key, k_result.labels)
        v_work, _ = _permute_by_labels(value, None, sorted_indices=k_order)
        return q_work, k_work, v_work, q_order, k_order

    (q_work, k_work, v_work, q_order, k_order), permutation_ms = _timed_cuda(
        do_permutation, enabled=config.measure_timing
    )

    def select_varlen():
        if method == "svg2_official_top_p":
            return top_p_map(
                centroid_scores,
                k_result.sizes,
                top_p=config.top_p,
                min_k_ratio=config.min_k_ratio,
            )
        if calibrated_map is None:
            raise RuntimeError("missing calibrated exact-budget map")
        return calibrated_map

    if method == "svg2_official_top_p":
        block_map, selection_ms = _timed_cuda(select_varlen, enabled=config.measure_timing)
    else:
        block_map = select_varlen()
        selection_ms = calibration_ms
    effective_density = config.density
    if method == "svg2_official_top_p":
        areas = q_result.sizes.long().unsqueeze(-1) * k_result.sizes.long().unsqueeze(-2)
        effective_density = float((areas * block_map).sum() / areas.sum())
        metadata.update(
            {
                "top_p": config.top_p,
                "min_k_ratio": config.min_k_ratio,
                "reported_actual_density_not_ranked_as_target_density": effective_density,
            }
        )
    plan = _plan_metrics(
        method=method,
        backend=config.backend,
        parameter_origin=config.parameter_origin,
        density=effective_density if method == "svg2_official_top_p" else config.density,
        block_map=block_map,
        q_sizes=q_result.sizes,
        k_sizes=k_result.sizes,
        q_sorted_indices=q_order,
        k_sorted_indices=k_order,
        cluster_ms=cluster_ms,
        permutation_ms=permutation_ms,
        selection_ms=selection_ms,
        metadata=metadata,
    )
    plan.metadata["original_length"] = query.shape[2]
    return q_work, k_work, v_work, plan


def _route_random_block(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    config: MethodConfig,
    state: RoutingState,
    layer: int,
    call_index: int,
):
    del state
    batch, heads, length, _ = query.shape
    blocks = math.ceil(length / config.block_size)
    generator = torch.Generator(device=query.device)
    generator.manual_seed(config.cluster_seed + layer * 1009 + call_index * 17)
    scores = torch.rand(
        (batch, heads, blocks, blocks),
        generator=generator,
        device=query.device,
        dtype=torch.float32,
    )
    return _fixed_plan(
        query,
        key,
        value,
        config=config,
        method="random_block",
        score_override=scores,
        metadata={"selector": "deterministic_random_block"},
    )


def _route_local_3d(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    config: MethodConfig,
    state: RoutingState,
    layer: int,
    call_index: int,
):
    del state, layer, call_index
    batch, heads, length, _ = query.shape
    blocks = math.ceil(length / config.block_size)
    frames = int(config.route_params.get("frames_latent", 21))
    height = int(config.route_params.get("height_latent", 30))
    width = int(config.route_params.get("width_latent", 52))
    if frames * height * width != length:
        raise ValueError(
            f"3D local layout {frames}x{height}x{width} does not match length {length}"
        )
    centers = (
        torch.arange(blocks, device=query.device) * config.block_size
        + config.block_size // 2
    ).clamp_max(length - 1)
    t = centers // (height * width)
    remainder = centers % (height * width)
    y = remainder // width
    x = remainder % width
    coordinates = torch.stack(
        (
            t.float() / max(1, frames - 1),
            y.float() / max(1, height - 1),
            x.float() / max(1, width - 1),
        ),
        dim=1,
    )
    weights = torch.tensor(
        config.route_params.get("distance_weights", [1.0, 1.0, 1.0]),
        device=query.device,
        dtype=torch.float32,
    )
    distance = ((coordinates[:, None] - coordinates[None, :]).abs() * weights).sum(dim=-1)
    scores = (-distance).view(1, 1, blocks, blocks).expand(batch, heads, -1, -1)
    return _fixed_plan(
        query,
        key,
        value,
        config=config,
        method="local_3d",
        score_override=scores,
        metadata={
            "selector": "normalized_3d_manhattan",
            "layout": [frames, height, width],
            "distance_weights": weights.tolist(),
        },
    )


def _local_qsort_labels(query: torch.Tensor, block_size: int, clusters: int) -> torch.Tensor:
    batch, heads, length, dim = query.shape
    blocks = math.ceil(length / block_size)
    padded = _pad_to_blocks(query.float(), block_size)
    vectors = padded.view(batch * heads * blocks, block_size, dim)
    normalized = F.normalize(vectors, dim=-1, eps=1e-12)
    initial = torch.linspace(0, block_size - 1, clusters, device=query.device).round().long()
    centroids = normalized.index_select(1, initial).contiguous()
    labels = torch.zeros(
        (vectors.shape[0], block_size), dtype=torch.long, device=query.device
    )
    for _ in range(int(3)):
        labels = torch.bmm(normalized, centroids.transpose(1, 2)).argmax(dim=-1)
        offsets = torch.arange(vectors.shape[0], device=query.device).view(-1, 1) * clusters
        flat_labels = (labels + offsets).reshape(-1)
        sums = torch.zeros(
            (vectors.shape[0] * clusters, dim), device=query.device, dtype=torch.float32
        )
        sums.index_add_(0, flat_labels, normalized.reshape(-1, dim))
        counts = torch.bincount(flat_labels, minlength=vectors.shape[0] * clusters)
        centroids = F.normalize(
            sums.view(vectors.shape[0], clusters, dim)
            / counts.view(vectors.shape[0], clusters, 1).clamp_min(1),
            dim=-1,
            eps=1e-12,
        )
    block_ids = torch.arange(blocks, device=query.device).view(1, 1, blocks, 1)
    labels = labels.view(batch, heads, blocks, block_size) + block_ids * clusters
    return labels.reshape(batch, heads, blocks * block_size)[..., :length]


def _route_qsort_local8(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    config: MethodConfig,
    state: RoutingState,
    layer: int,
    call_index: int,
):
    del state, layer, call_index
    labels, cluster_ms = _timed_cuda(
        lambda: _local_qsort_labels(query, config.block_size, 8),
        enabled=config.measure_timing,
    )
    return _fixed_plan(
        query,
        key,
        value,
        config=config,
        method="qsort_local8",
        q_labels=labels,
        cluster_ms=cluster_ms,
        metadata={"selector": "local_q_kmeans8_then_block", "counted_as_clustering": False},
    )


def route_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    config: MethodConfig,
    state: RoutingState,
    layer: int,
    call_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RoutePlan]:
    if config.method == "random_block":
        return _route_random_block(
            query, key, value, config=config, state=state, layer=layer, call_index=call_index
        )
    if config.method == "local_3d":
        return _route_local_3d(
            query, key, value, config=config, state=state, layer=layer, call_index=call_index
        )
    if config.method == "qsort_local8":
        return _route_qsort_local8(
            query, key, value, config=config, state=state, layer=layer, call_index=call_index
        )
    if config.method in {
        "capacity_balanced",
        "radius_adaptive",
        "hierarchical",
        "product_quantized",
        "spatiotemporal",
        "query_metric",
    }:
        from .routes import self_cluster

        function = {
            "capacity_balanced": self_cluster.route_capacity_balanced,
            "radius_adaptive": self_cluster.route_radius_adaptive,
            "hierarchical": self_cluster.route_hierarchical,
            "product_quantized": self_cluster.route_product_quantized,
            "spatiotemporal": self_cluster.route_spatiotemporal,
            "query_metric": self_cluster.route_query_metric,
        }[config.method]
        return function(
            query,
            key,
            value,
            config=config,
            state=state,
            layer=layer,
            call_index=call_index,
        )
    if config.method in {"adacluster", "svoo", "scope"}:
        from .routes import papers

        function = {
            "adacluster": papers.route_adacluster,
            "svoo": papers.route_svoo,
            "scope": papers.route_scope,
        }[config.method]
        return function(
            query,
            key,
            value,
            config=config,
            state=state,
            layer=layer,
            call_index=call_index,
        )
    return _route_attention_legacy(
        query,
        key,
        value,
        config=config,
        state=state,
        layer=layer,
        call_index=call_index,
    )
