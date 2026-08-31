"""Exact-budget history selectors for fixed blocks and variable clusters."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from .config import SparseHistoryConfig
from .methods import method_spec
from .stats import TimingBreakdown


INDEXED_PRETRANSFER_METHODS = {
    "block64_history",
    "kcluster32_history",
    "fixed_k128_history",
    "fixed_k256_history",
    "qlocal_kmeans8_ar",
    "radius_k256_ar",
    "temporal_k256_t16_ar",
    "coverage_cluster_history",
    "vaware_cluster_history",
    "transfer_vaware_hybrid_history",
}

SUMMARY_PRETRANSFER_METHODS = {
    "coverage_cluster_history",
    "vaware_cluster_history",
    "transfer_vaware_hybrid_history",
}


@dataclass
class FrameIndex:
    frame_id: int
    spatial_height: int
    spatial_width: int
    key: torch.Tensor
    value: torch.Tensor
    block_centroids: torch.Tensor
    block_value_centroids: torch.Tensor
    block_cluster_membership: torch.Tensor
    block_starts: torch.Tensor
    block_ends: torch.Tensor
    cluster_centroids: torch.Tensor
    cluster_labels: torch.Tensor
    cluster_counts: torch.Tensor
    cluster_radii: torch.Tensor
    index_bytes: int
    routing_bytes: int
    archive_bytes: int
    index_elapsed_s: float


@dataclass
class PretransferQuerySummary:
    query_labels: torch.Tensor
    query_centroids: torch.Tensor
    query_group_sizes: torch.Tensor
    query_tokens: int
    summary_bytes: int
    q_summary_s: float
    d2h_s: float


@dataclass
class SparseSelection:
    frame_ids: torch.Tensor
    token_ids: torch.Tensor
    scores: torch.Tensor
    candidate_history_tokens: int
    selected_history_tokens: int
    candidate_units: int
    selected_units: int
    cluster_size_min: int | None
    cluster_size_max: int | None
    index_bytes: int
    timing: TimingBreakdown


def _tensor_bytes(*tensors: torch.Tensor) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def _query_block_means(query: torch.Tensor, block_size: int) -> torch.Tensor:
    if query.ndim != 4:
        raise ValueError("query must be [B,Q,H,D]")
    batch, tokens, heads, dim = query.shape
    blocks = []
    for start in range(0, tokens, block_size):
        blocks.append(query[:, start : start + block_size].float().mean(dim=1))
    return torch.stack(blocks, dim=2).reshape(batch, heads, -1, dim)


def summarize_query_for_pretransfer(
    query: torch.Tensor, block_size: int
) -> PretransferQuerySummary:
    """Summarize Q on its source device and transfer only compact prototypes."""

    if query.ndim != 4:
        raise ValueError("query must be [B,Q,H,D]")
    if query.is_cuda:
        torch.cuda.synchronize(query.device)
    summary_start = time.perf_counter()
    centroids = _query_block_means(query, block_size)
    if query.is_cuda:
        torch.cuda.synchronize(query.device)
    q_summary_s = time.perf_counter() - summary_start

    transfer_start = time.perf_counter()
    centroids_cpu = centroids.detach().to("cpu")
    if query.is_cuda:
        torch.cuda.synchronize(query.device)
    d2h_s = time.perf_counter() - transfer_start

    batch, query_tokens, heads, _ = query.shape
    labels = torch.div(
        torch.arange(query_tokens, dtype=torch.long),
        block_size,
        rounding_mode="floor",
    ).view(1, 1, query_tokens).expand(batch, heads, -1).clone()
    groups = centroids_cpu.shape[2]
    group_sizes = torch.zeros((batch, heads, groups), dtype=torch.long)
    group_sizes.scatter_add_(2, labels, torch.ones_like(labels))
    return PretransferQuerySummary(
        query_labels=labels,
        query_centroids=centroids_cpu,
        query_group_sizes=group_sizes,
        query_tokens=query_tokens,
        summary_bytes=_tensor_bytes(centroids_cpu),
        q_summary_s=q_summary_s,
        d2h_s=d2h_s,
    )


def _batched_spherical_kmeans(
    vectors: torch.Tensor,
    clusters: int,
    *,
    iterations: int,
    tolerance: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent deterministic spherical K-means for ``[B,H,T,D]``."""

    if vectors.ndim != 4:
        raise ValueError("vectors must be [B,H,T,D]")
    batch, heads, tokens, dim = vectors.shape
    clusters = min(int(clusters), tokens)
    flat = F.normalize(vectors.float().reshape(batch * heads, tokens, dim), dim=-1)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    initial = torch.randperm(tokens, generator=generator)[:clusters].to(flat.device)
    centroids = flat.index_select(1, initial).clone()
    labels = torch.full((batch * heads, tokens), -1, dtype=torch.long, device=flat.device)

    for iteration in range(iterations):
        previous = labels.clone()
        similarities = torch.bmm(flat, centroids.transpose(1, 2))
        best_similarity, labels = similarities.max(dim=2)
        counts = torch.stack(
            [
                torch.bincount(row, minlength=clusters)
                for row in labels
            ]
        )
        for row_index in range(batch * heads):
            empty_clusters = torch.nonzero(
                counts[row_index] == 0, as_tuple=False
            ).flatten()
            if not empty_clusters.numel():
                continue
            candidates = torch.argsort(best_similarity[row_index], stable=True)
            used: set[int] = set()
            for empty_cluster in empty_clusters.tolist():
                source_token = None
                for candidate in candidates.tolist():
                    source_cluster = int(labels[row_index, candidate])
                    if candidate not in used and int(counts[row_index, source_cluster]) > 1:
                        source_token = candidate
                        break
                if source_token is None:
                    raise RuntimeError("unable to repair empty K-means cluster")
                used.add(source_token)
                source_cluster = int(labels[row_index, source_token])
                counts[row_index, source_cluster] -= 1
                counts[row_index, empty_cluster] = 1
                labels[row_index, source_token] = empty_cluster

        offsets = torch.arange(batch * heads, device=flat.device).view(-1, 1) * clusters
        flat_labels = (labels + offsets).reshape(-1)
        sums = torch.zeros(
            (batch * heads * clusters, dim), dtype=torch.float32, device=flat.device
        )
        sums.index_add_(0, flat_labels, flat.reshape(-1, dim))
        counts = torch.bincount(flat_labels, minlength=batch * heads * clusters).view(
            batch * heads, clusters
        )
        new_centroids = F.normalize(
            sums.view(batch * heads, clusters, dim) / counts.clamp_min(1).unsqueeze(-1),
            dim=-1,
        )
        shift = (1.0 - (centroids * new_centroids).sum(dim=-1)).abs().amax()
        centroids = new_centroids
        if iteration > 0 and (torch.equal(labels, previous) or float(shift) <= tolerance):
            break

    return (
        centroids.view(batch, heads, clusters, dim),
        labels.view(batch, heads, tokens),
        counts.view(batch, heads, clusters),
    )


def build_frame_index(
    frame_id: int,
    key_for_index: torch.Tensor,
    value_storage: torch.Tensor,
    key_storage: torch.Tensor,
    config: SparseHistoryConfig,
    *,
    spatial_height: int,
    spatial_width: int,
) -> FrameIndex:
    """Build both fixed-block and KCluster32 metadata for one archived frame."""

    if key_for_index.ndim != 4:
        raise ValueError("frame key must be [B,T,H,D]")
    if value_storage.shape != key_storage.shape or key_storage.shape != key_for_index.shape:
        raise ValueError("key/value storage and index tensors must share [B,T,H,D]")
    start_time = time.perf_counter()
    key_bhtd = key_for_index.permute(0, 2, 1, 3)
    _, _, tokens, _ = key_bhtd.shape

    if config.method == "block64_history" or config.method in SUMMARY_PRETRANSFER_METHODS:
        block_centroids = []
        block_value_centroids = []
        starts = []
        ends = []
        value_bhtd = value_storage.permute(0, 2, 1, 3)
        for start in range(0, tokens, config.block_size):
            end = min(start + config.block_size, tokens)
            block_centroids.append(key_bhtd[:, :, start:end].float().mean(dim=2))
            block_value_centroids.append(
                value_bhtd[:, :, start:end].float().mean(dim=2)
            )
            starts.append(start)
            ends.append(end)
        block_centroids_tensor = torch.stack(block_centroids, dim=2)
        block_value_centroids_tensor = torch.stack(block_value_centroids, dim=2)
        block_starts = torch.tensor(starts, dtype=torch.long, device=key_for_index.device)
        block_ends = torch.tensor(ends, dtype=torch.long, device=key_for_index.device)
    else:
        block_centroids_tensor = torch.empty(
            (*key_bhtd.shape[:2], 0, key_bhtd.shape[-1]),
            dtype=torch.float32,
            device=key_for_index.device,
        )
        block_value_centroids_tensor = torch.empty_like(block_centroids_tensor)
        block_starts = torch.empty(0, dtype=torch.long, device=key_for_index.device)
        block_ends = torch.empty(0, dtype=torch.long, device=key_for_index.device)

    if config.method in INDEXED_PRETRANSFER_METHODS - {"block64_history"}:
        spec = method_spec(config.method)
        if config.method_params:
            spec = replace(spec, **config.method_params)
        clusters = (
            config.clusters_per_frame
            if config.method == "kcluster32_history"
            else (
                spec.remote_clusters
                if config.method in SUMMARY_PRETRANSFER_METHODS
                and spec.remote_clusters is not None
                else spec.k_clusters
            )
        )
        cluster_centroids, cluster_labels, cluster_counts = _batched_spherical_kmeans(
            key_bhtd,
            clusters,
            iterations=spec.iterations,
            tolerance=config.kmeans_tolerance,
            seed=config.seed + int(frame_id),
        )
        normalized = F.normalize(key_bhtd.float(), dim=-1)
        assigned = cluster_centroids.gather(
            2,
            cluster_labels.unsqueeze(-1).expand(-1, -1, -1, cluster_centroids.shape[-1]),
        )
        residual = 1.0 - (normalized * assigned).sum(dim=-1)
        flat_labels = cluster_labels.reshape(-1, tokens)
        flat_residual = residual.reshape(-1, tokens)
        flat_radii = torch.zeros(
            (flat_labels.shape[0], cluster_centroids.shape[2]),
            dtype=torch.float32,
            device=key_for_index.device,
        )
        flat_radii.scatter_reduce_(
            1, flat_labels, flat_residual, reduce="amax", include_self=True
        )
        cluster_radii = flat_radii.view(*cluster_counts.shape)
        if config.method in SUMMARY_PRETRANSFER_METHODS:
            padded_tokens = math.ceil(tokens / config.block_size) * config.block_size
            labels_padded = F.pad(
                cluster_labels,
                (0, padded_tokens - tokens),
                value=cluster_centroids.shape[2],
            )
            membership = F.one_hot(
                labels_padded,
                num_classes=cluster_centroids.shape[2] + 1,
            )[..., : cluster_centroids.shape[2]]
            membership = membership.view(
                *cluster_labels.shape[:2],
                padded_tokens // config.block_size,
                config.block_size,
                cluster_centroids.shape[2],
            ).float().sum(dim=3)
            valid = torch.full(
                (padded_tokens // config.block_size,),
                config.block_size,
                dtype=torch.float32,
                device=membership.device,
            )
            if tokens % config.block_size:
                valid[-1] = tokens % config.block_size
            block_cluster_membership = membership / valid.view(1, 1, -1, 1)
        else:
            block_cluster_membership = torch.empty(
                (*key_bhtd.shape[:2], 0, 0),
                dtype=torch.float32,
                device=key_for_index.device,
            )
    else:
        cluster_centroids = torch.empty(
            (*key_bhtd.shape[:2], 0, key_bhtd.shape[-1]),
            dtype=torch.float32,
            device=key_for_index.device,
        )
        cluster_labels = torch.empty(
            (*key_bhtd.shape[:2], tokens), dtype=torch.long, device=key_for_index.device
        )
        cluster_counts = torch.empty(
            (*key_bhtd.shape[:2], 0), dtype=torch.long, device=key_for_index.device
        )
        cluster_radii = torch.empty(
            (*key_bhtd.shape[:2], 0), dtype=torch.float32, device=key_for_index.device
        )
        block_cluster_membership = torch.empty(
            (*key_bhtd.shape[:2], 0, 0),
            dtype=torch.float32,
            device=key_for_index.device,
        )
    index_elapsed = time.perf_counter() - start_time

    metadata = (
        block_centroids_tensor.detach().cpu(),
        block_value_centroids_tensor.detach().cpu(),
        block_cluster_membership.detach().cpu(),
        block_starts.detach().cpu(),
        block_ends.detach().cpu(),
        cluster_centroids.detach().cpu(),
        cluster_labels.detach().cpu(),
        cluster_counts.detach().cpu(),
        cluster_radii.detach().cpu(),
    )
    total_index_bytes = _tensor_bytes(*metadata)
    if config.method == "block64_history":
        routing_bytes = _tensor_bytes(metadata[0])
    elif config.method in SUMMARY_PRETRANSFER_METHODS:
        routing_bytes = _tensor_bytes(
            metadata[0], metadata[1], metadata[2], metadata[5], metadata[7]
        )
    elif config.method in INDEXED_PRETRANSFER_METHODS:
        routing_bytes = _tensor_bytes(metadata[5], metadata[7], metadata[8])
    else:
        routing_bytes = 0
    return FrameIndex(
        frame_id=int(frame_id),
        spatial_height=int(spatial_height),
        spatial_width=int(spatial_width),
        key=key_storage,
        value=value_storage,
        block_centroids=metadata[0],
        block_value_centroids=metadata[1],
        block_cluster_membership=metadata[2],
        block_starts=metadata[3],
        block_ends=metadata[4],
        cluster_centroids=metadata[5],
        cluster_labels=metadata[6],
        cluster_counts=metadata[7],
        cluster_radii=metadata[8],
        index_bytes=total_index_bytes,
        routing_bytes=routing_bytes,
        archive_bytes=_tensor_bytes(key_storage, value_storage),
        index_elapsed_s=index_elapsed,
    )


def _exact_budget(candidate_tokens: int, density: float) -> int:
    if candidate_tokens < 1:
        return 0
    return max(1, min(candidate_tokens, int(round(candidate_tokens * density))))


def _sort_selected(
    frame_ids: list[int], token_ids: list[int], scores: list[float]
) -> tuple[list[int], list[int], list[float]]:
    order = sorted(range(len(frame_ids)), key=lambda index: (frame_ids[index], token_ids[index]))
    return (
        [frame_ids[index] for index in order],
        [token_ids[index] for index in order],
        [scores[index] for index in order],
    )


def _indexed_query_labels(query: torch.Tensor, config: SparseHistoryConfig) -> torch.Tensor:
    batch, tokens, heads, _ = query.shape
    if config.method in {
        "kcluster32_history",
        "fixed_k128_history",
        "fixed_k256_history",
    }:
        return torch.zeros((batch, heads, tokens), dtype=torch.long, device=query.device)
    if config.method == "qlocal_kmeans8_ar":
        query_bhtd = query.permute(0, 2, 1, 3)
        labels = torch.empty((batch, heads, tokens), dtype=torch.long, device=query.device)
        offset = 0
        for block_id, start in enumerate(range(0, tokens, 64)):
            part = query_bhtd[:, :, start : start + 64]
            _, local, _ = _batched_spherical_kmeans(
                part,
                min(8, part.shape[2]),
                iterations=3,
                tolerance=config.kmeans_tolerance,
                seed=config.seed + block_id * 97,
            )
            labels[:, :, start : start + part.shape[2]] = local + offset
            offset += int(local.max()) + 1
        return labels
    base = torch.div(
        torch.arange(tokens, device=query.device), 64, rounding_mode="floor"
    )
    return base.view(1, 1, tokens).expand(batch, heads, -1).clone()


def _indexed_group_means(
    query: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    query_bhtd = query.permute(0, 2, 1, 3).float()
    groups = int(labels.max()) + 1
    sums = torch.zeros(
        (*query_bhtd.shape[:2], groups, query_bhtd.shape[-1]),
        dtype=torch.float32,
        device=query.device,
    )
    sums.scatter_add_(
        2,
        labels.unsqueeze(-1).expand(-1, -1, -1, query_bhtd.shape[-1]),
        query_bhtd,
    )
    counts = torch.zeros(
        (*query_bhtd.shape[:2], groups), dtype=torch.long, device=query.device
    )
    counts.scatter_add_(2, labels, torch.ones_like(labels))
    return sums / counts.clamp_min(1).unsqueeze(-1), counts


def _standardize_last(scores: torch.Tensor) -> torch.Tensor:
    mean = scores.mean(dim=-1, keepdim=True)
    scale = scores.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (scores - mean) / scale


def _tier_token_counts(
    budget: int, base_fraction: float, local_fraction: float
) -> tuple[int, int, int]:
    if base_fraction < 0 or local_fraction < 0 or base_fraction + local_fraction > 1:
        raise ValueError("invalid proposed-method budget fractions")
    base = max(1, min(budget, int(round(budget * base_fraction))))
    local = max(0, min(budget - base, int(round(budget * local_fraction))))
    return base, local, budget - base - local


def _expand_tiered_block_orders(
    entries: list[tuple[int, int, int]],
    frame_tokens: int,
    orders: tuple[list[int], list[int], list[int]],
    tier_counts: tuple[int, int, int],
    *,
    allowed_tokens: set[int] | None = None,
) -> torch.Tensor:
    selected: list[int] = []
    selected_set: set[int] = set()

    def add_until(order: list[int], target: int) -> None:
        if len(selected) >= target:
            return
        for unit in order:
            frame_index, start, end = entries[unit]
            for token in range(start, end):
                flat = frame_index * frame_tokens + token
                if flat in selected_set:
                    continue
                if allowed_tokens is not None and flat not in allowed_tokens:
                    continue
                selected.append(flat)
                selected_set.add(flat)
                if len(selected) == target:
                    return

    base, local, remote = tier_counts
    add_until(orders[0], base)
    add_until(orders[1], base + local)
    add_until(orders[2], base + local + remote)
    if len(selected) < sum(tier_counts):
        fallback = list(dict.fromkeys((*orders[0], *orders[1], *orders[2])))
        add_until(fallback, sum(tier_counts))
    if len(selected) != sum(tier_counts):
        raise RuntimeError(
            f"tiered block selection produced {len(selected)} tokens instead of {sum(tier_counts)}"
        )
    return torch.tensor(sorted(selected), dtype=torch.long)


def _proposed_indexed_route(
    summary: PretransferQuerySummary,
    frames: list[FrameIndex],
    config: SparseHistoryConfig,
    *,
    exact_k_tokens: int,
):
    """Coverage/V-aware routing using only Q summaries and CPU K/V prototypes."""

    from .ar_routing import build_route_plan

    spec = method_spec(config.method)
    if config.method_params:
        spec = replace(spec, **config.method_params)
    query_centroids = summary.query_centroids.float()
    query_labels = summary.query_labels
    batch, heads, groups, dim = query_centroids.shape
    frame_tokens = frames[0].key.shape[1]
    candidate_tokens = len(frames) * frame_tokens
    budget = _exact_budget(candidate_tokens, config.history_density)
    base_fraction = float(spec.base_fraction or 0.80)
    local_fraction = float(spec.local_fraction or 0.10)
    tier_counts = _tier_token_counts(budget, base_fraction, local_fraction)

    entries: list[tuple[int, int, int]] = []
    direct_parts = []
    remote_parts = []
    value_norm_parts = []
    block_frame_ids: list[int] = []
    block_centers: list[int] = []
    for frame_index, frame in enumerate(frames):
        key_blocks = frame.block_centroids.float()
        value_blocks = frame.block_value_centroids.float()
        direct_parts.append(
            torch.einsum("bhqd,bhkd->bhqk", query_centroids, key_blocks)
            / math.sqrt(dim)
        )
        q_to_cluster = torch.einsum(
            "bhqd,bhcd->bhqc",
            query_centroids,
            frame.cluster_centroids.float(),
        ) / math.sqrt(dim)
        remote_parts.append(
            torch.einsum(
                "bhqc,bhkc->bhqk",
                q_to_cluster,
                frame.block_cluster_membership.float(),
            )
        )
        value_norm_parts.append(torch.linalg.vector_norm(value_blocks, dim=-1))
        for start, end in zip(frame.block_starts.tolist(), frame.block_ends.tolist()):
            entries.append((frame_index, int(start), int(end)))
            block_frame_ids.append(frame.frame_id)
            block_centers.append(min(frame_tokens - 1, (int(start) + int(end) - 1) // 2))

    direct_scores = torch.cat(direct_parts, dim=-1)
    cluster_scores = torch.cat(remote_parts, dim=-1)
    value_norms = torch.cat(value_norm_parts, dim=-1).unsqueeze(2)
    probability_proxy = torch.softmax(direct_scores, dim=-1)
    prototype_value_score = probability_proxy * value_norms
    if config.method == "coverage_cluster_history":
        remote_scores = cluster_scores
        online_proxy = "q_summary_to_k_cluster"
    else:
        remote_scores = _standardize_last(cluster_scores) + float(
            spec.v_weight or 0.75
        ) * _standardize_last(prototype_value_score)
        online_proxy = "q_summary_k_proxy_plus_probability_times_v_prototype_norm"

    height = frames[0].spatial_height
    width = frames[0].spatial_width
    if height * width != frame_tokens:
        raise ValueError("frame spatial geometry does not match token count")
    query_centers = (
        torch.arange(groups, dtype=torch.long) * config.block_size
        + config.block_size // 2
    ).clamp_max(summary.query_tokens - 1)
    query_spatial = query_centers.remainder(frame_tokens)
    query_y = query_spatial // width
    query_x = query_spatial % width
    history_center = torch.tensor(block_centers, dtype=torch.long)
    history_y = history_center // width
    history_x = history_center % width
    dy = (query_y[:, None] - history_y[None, :]).abs().float() / max(1, height - 1)
    dx = (query_x[:, None] - history_x[None, :]).abs().float() / max(1, width - 1)
    newest_frame = max(block_frame_ids)
    frame_age = torch.tensor(
        [newest_frame - frame_id for frame_id in block_frame_ids], dtype=torch.float32
    )
    age_scale = max(1.0, float(frame_age.max()))
    local_scores = -(2.0 * frame_age.view(1, -1) / age_scale + dy + dx)
    remote_min_frames = int(spec.remote_min_frames or 2)
    remote_mask = frame_age >= remote_min_frames
    if not bool(remote_mask.any()):
        remote_mask = torch.ones_like(remote_mask, dtype=torch.bool)

    selections: list[list[list[torch.Tensor]]] = [
        [[] for _ in range(heads)] for _ in range(batch)
    ]
    planned_union_sizes = []
    for batch_index in range(batch):
        for head in range(heads):
            allowed_tokens: set[int] | None = None
            if config.method == "transfer_vaware_hybrid_history":
                transfer_budget = max(
                    budget,
                    min(
                        candidate_tokens,
                        int(
                            round(
                                candidate_tokens
                                * config.history_density
                                * float(spec.transfer_multiplier or 1.25)
                            )
                        ),
                    ),
                )
                pool_tiers = _tier_token_counts(
                    transfer_budget, base_fraction, local_fraction
                )
                base_global = direct_scores[batch_index, head].amax(dim=0)
                local_global = local_scores.amax(dim=0)
                remote_global = remote_scores[batch_index, head].amax(dim=0)
                remote_global = remote_global.masked_fill(~remote_mask, -float("inf"))
                pool = _expand_tiered_block_orders(
                    entries,
                    frame_tokens,
                    (
                        torch.argsort(base_global, descending=True, stable=True).tolist(),
                        torch.argsort(local_global, descending=True, stable=True).tolist(),
                        torch.argsort(remote_global, descending=True, stable=True).tolist(),
                    ),
                    pool_tiers,
                )
                allowed_tokens = set(pool.tolist())
                planned_union_sizes.append(len(allowed_tokens))
            for group in range(groups):
                remote_row = remote_scores[batch_index, head, group].masked_fill(
                    ~remote_mask, -float("inf")
                )
                selected = _expand_tiered_block_orders(
                    entries,
                    frame_tokens,
                    (
                        torch.argsort(
                            direct_scores[batch_index, head, group],
                            descending=True,
                            stable=True,
                        ).tolist(),
                        torch.argsort(
                            local_scores[group], descending=True, stable=True
                        ).tolist(),
                        torch.argsort(
                            remote_row, descending=True, stable=True
                        ).tolist(),
                    ),
                    tier_counts,
                    allowed_tokens=allowed_tokens,
                )
                selections[batch_index][head].append(selected)

    history_frames = torch.tensor(
        [frame.frame_id for frame in frames], dtype=torch.long
    ).repeat_interleave(frame_tokens)
    history_tokens = torch.arange(frame_tokens, dtype=torch.long).repeat(len(frames))
    history_frames = history_frames.view(1, 1, -1).expand(batch, heads, -1)
    history_tokens = history_tokens.view(1, 1, -1).expand(batch, heads, -1)
    metadata = {
        "parameter_origin": spec.parameter_origin,
        "index_source": "per_frame_cpu_kv_prototypes",
        "query_summary_only": True,
        "query_summary_bytes": summary.summary_bytes,
        "routing_index_bytes": sum(frame.routing_bytes for frame in frames),
        "online_proxy": online_proxy,
        "output_residual_online": False,
        "output_residual_role": "offline_teacher_only",
        "base_fraction_candidate": base_fraction,
        "local_fraction_candidate": local_fraction,
        "remote_fraction_candidate": 1.0 - base_fraction - local_fraction,
        "base_tokens_per_group": tier_counts[0],
        "local_tokens_per_group": tier_counts[1],
        "remote_tokens_per_group": tier_counts[2],
        "remote_clusters_per_frame": int(spec.remote_clusters or spec.k_clusters),
        "remote_min_frames": remote_min_frames,
        "transfer_multiplier_candidate": (
            float(spec.transfer_multiplier or 1.25)
            if config.method == "transfer_vaware_hybrid_history"
            else None
        ),
        "planned_union_tokens_min": (
            min(planned_union_sizes) if planned_union_sizes else None
        ),
        "planned_union_tokens_max": (
            max(planned_union_sizes) if planned_union_sizes else None
        ),
        "preserves_original_token_order": True,
        "executes_original_kv": True,
    }
    return build_route_plan(
        method=config.method,
        routing_stage=config.routing_stage,
        query_labels=query_labels,
        selections=selections,
        history_frame_ids=history_frames,
        history_token_ids=history_tokens,
        candidate_history_tokens=candidate_tokens,
        exact_k_tokens=exact_k_tokens,
        density=config.history_density,
        metadata=metadata,
    )


def route_indexed_history(
    query: torch.Tensor | PretransferQuerySummary,
    frames: list[FrameIndex],
    config: SparseHistoryConfig,
    *,
    exact_k_tokens: int,
):
    """Build a per-query-group route from per-frame archive indices."""

    if config.method not in INDEXED_PRETRANSFER_METHODS:
        raise ValueError(f"method is not supported by indexed routing: {config.method}")
    if not frames:
        raise ValueError("indexed routing requires candidate frames")
    from .ar_routing import build_route_plan

    if isinstance(query, PretransferQuerySummary):
        if config.method not in SUMMARY_PRETRANSFER_METHODS:
            raise ValueError(
                f"compact query summaries are unsupported for {config.method}"
            )
        query_labels = query.query_labels
        query_centroids = query.query_centroids
        batch, heads, _, dim = query_centroids.shape
        query_tokens = query.query_tokens
        query_device = query_centroids.device
    else:
        batch, query_tokens, heads, dim = query.shape
        query_labels = _indexed_query_labels(query, config)
        query_centroids, _ = _indexed_group_means(query, query_labels)
        query_device = query.device
    frame_tokens = frames[0].key.shape[1]
    candidate_tokens = len(frames) * frame_tokens
    budget = _exact_budget(candidate_tokens, config.history_density)
    selections: list[list[list[torch.Tensor]]] = [
        [[] for _ in range(heads)] for _ in range(batch)
    ]
    metadata = {
        "parameter_origin": method_spec(config.method).parameter_origin,
        "index_source": "per_frame_archive",
        "routing_index_bytes": sum(frame.routing_bytes for frame in frames),
    }
    if budget == candidate_tokens:
        full = torch.arange(candidate_tokens, dtype=torch.long, device=query_device)
        for batch_index in range(batch):
            for head in range(heads):
                groups = int(query_labels[batch_index, head].max()) + 1
                selections[batch_index][head] = [full for _ in range(groups)]
        history_frames = torch.tensor(
            [frame.frame_id for frame in frames],
            dtype=torch.long,
            device=query_device,
        ).repeat_interleave(frame_tokens)
        history_tokens = torch.arange(
            frame_tokens, dtype=torch.long, device=query_device
        ).repeat(len(frames))
        history_frames = history_frames.view(1, 1, -1).expand(batch, heads, -1)
        history_tokens = history_tokens.view(1, 1, -1).expand(batch, heads, -1)
        metadata["full_density_fast_path"] = True
        return build_route_plan(
            method=config.method,
            routing_stage=config.routing_stage,
            query_labels=query_labels,
            selections=selections,
            history_frame_ids=history_frames,
            history_token_ids=history_tokens,
            candidate_history_tokens=candidate_tokens,
            exact_k_tokens=exact_k_tokens,
            density=config.history_density,
            metadata=metadata,
        )

    if config.method in SUMMARY_PRETRANSFER_METHODS:
        if not isinstance(query, PretransferQuerySummary):
            raise ValueError(
                f"{config.method} requires a compact pre-transfer query summary"
            )
        return _proposed_indexed_route(
            query,
            frames,
            config,
            exact_k_tokens=exact_k_tokens,
        )

    if config.method == "block64_history":
        entries: list[tuple[int, int, int]] = []
        centroids = []
        for frame_index, frame in enumerate(frames):
            centroids.append(frame.block_centroids.to(query_device))
            entries.extend(
                (frame_index, int(start), int(end))
                for start, end in zip(frame.block_starts, frame.block_ends)
            )
        all_centroids = torch.cat(centroids, dim=2)
        scores = torch.einsum(
            "bhqd,bhkd->bhqk", query_centroids, all_centroids.float()
        ) / math.sqrt(dim)
        for batch_index in range(batch):
            for head in range(heads):
                for group in range(query_centroids.shape[2]):
                    selected = []
                    order = torch.argsort(
                        scores[batch_index, head, group],
                        descending=True,
                        stable=True,
                    )
                    for unit in order.tolist():
                        frame_index, start, end = entries[unit]
                        take = min(end - start, budget - len(selected))
                        selected.extend(
                            frame_index * frame_tokens + token
                            for token in range(start, start + take)
                        )
                        if len(selected) == budget:
                            break
                    selections[batch_index][head].append(
                        torch.tensor(sorted(selected), dtype=torch.long, device=query_device)
                    )
    else:
        cluster_entries: list[tuple[int, int]] = []
        centroids, counts, radii = [], [], []
        cluster_frame_indices = []
        for frame_index, frame in enumerate(frames):
            frame_centroids = frame.cluster_centroids.to(query_device)
            centroids.append(frame_centroids)
            counts.append(frame.cluster_counts.to(query_device))
            radii.append(frame.cluster_radii.to(query_device))
            cluster_entries.extend(
                (frame_index, cluster_index)
                for cluster_index in range(frame_centroids.shape[2])
            )
            cluster_frame_indices.extend([frame_index] * frame_centroids.shape[2])
        all_centroids = torch.cat(centroids, dim=2).float()
        all_counts = torch.cat(counts, dim=2)
        all_radii = torch.cat(radii, dim=2)
        scores = torch.einsum(
            "bhqd,bhkd->bhqk", query_centroids, all_centroids
        ) / math.sqrt(dim)
        scores += all_counts.float().clamp_min(1).log().unsqueeze(2)
        if config.method == "radius_k256_ar":
            beta = float(method_spec(config.method).threshold or 0.5)
            scores += beta * all_radii.unsqueeze(2)
        temporal_bins: list[list[int]] | None = None
        if config.method == "temporal_k256_t16_ar":
            bin_count = min(method_spec(config.method).temporal_bins or 16, len(frames))
            temporal_bins = [[] for _ in range(bin_count)]
            for cluster_index, frame_index in enumerate(cluster_frame_indices):
                time_bin = min(bin_count - 1, frame_index * bin_count // len(frames))
                temporal_bins[time_bin].append(cluster_index)
        for batch_index in range(batch):
            for head in range(heads):
                for group in range(query_centroids.shape[2]):
                    score_row = scores[batch_index, head, group]
                    if temporal_bins is None:
                        order = torch.argsort(
                            score_row, descending=True, stable=True
                        ).tolist()
                    else:
                        per_bin = [
                            sorted(indices, key=lambda index: (-float(score_row[index]), index))
                            for indices in temporal_bins
                        ]
                        order = []
                        cursor = 0
                        while any(cursor < len(indices) for indices in per_bin):
                            for indices in per_bin:
                                if cursor < len(indices):
                                    order.append(indices[cursor])
                            cursor += 1
                    selected = []
                    for unit in order:
                        frame_index, cluster_index = cluster_entries[unit]
                        members = torch.nonzero(
                            frames[frame_index].cluster_labels[batch_index, head]
                            == cluster_index,
                            as_tuple=False,
                        ).flatten()
                        take = min(members.numel(), budget - len(selected))
                        selected.extend(
                            frame_index * frame_tokens + int(token)
                            for token in members[:take]
                        )
                        if len(selected) == budget:
                            break
                    selections[batch_index][head].append(
                        torch.tensor(sorted(selected), dtype=torch.long, device=query_device)
                    )
        metadata["cluster_size_min"] = int(all_counts.min())
        metadata["cluster_size_max"] = int(all_counts.max())

    history_frames = torch.tensor(
        [frame.frame_id for frame in frames], dtype=torch.long, device=query_device
    ).repeat_interleave(frame_tokens)
    history_tokens = torch.arange(
        frame_tokens, dtype=torch.long, device=query_device
    ).repeat(len(frames))
    history_frames = history_frames.view(1, 1, -1).expand(batch, heads, -1)
    history_tokens = history_tokens.view(1, 1, -1).expand(batch, heads, -1)
    return build_route_plan(
        method=config.method,
        routing_stage=config.routing_stage,
        query_labels=query_labels,
        selections=selections,
        history_frame_ids=history_frames,
        history_token_ids=history_tokens,
        candidate_history_tokens=candidate_tokens,
        exact_k_tokens=exact_k_tokens,
        density=config.history_density,
        metadata=metadata,
    )


def select_history(
    query: torch.Tensor,
    frames: list[FrameIndex],
    config: SparseHistoryConfig,
) -> SparseSelection:
    """Select original token coordinates from the supplied coarse frame pool."""

    if not frames:
        raise ValueError("at least one coarse-retrieved history frame is required")
    if query.ndim != 4:
        raise ValueError("query must be [B,Q,H,D]")
    batch, _, heads, dim = query.shape
    frame_tokens = frames[0].key.shape[1]
    if any(frame.key.shape != frames[0].key.shape for frame in frames):
        raise ValueError("all candidate frames must share the same K/V shape")
    candidate_tokens = len(frames) * frame_tokens
    budget = _exact_budget(candidate_tokens, config.history_density)
    timing = TimingBreakdown()

    q_start = time.perf_counter()
    q_blocks = _query_block_means(query, config.block_size)
    timing.q_summary_s = time.perf_counter() - q_start
    route_start = time.perf_counter()

    output_frames = torch.empty((batch, heads, budget), dtype=torch.long)
    output_tokens = torch.empty_like(output_frames)
    output_scores = torch.empty((batch, heads, budget), dtype=torch.float32)
    selected_units = 0
    candidate_units = 0
    cluster_min: int | None = None
    cluster_max: int | None = None

    if config.is_dense:
        dense_frames = []
        dense_tokens = []
        for frame in frames:
            dense_frames.extend([frame.frame_id] * frame_tokens)
            dense_tokens.extend(range(frame_tokens))
        frame_tensor = torch.tensor(dense_frames, dtype=torch.long)
        token_tensor = torch.tensor(dense_tokens, dtype=torch.long)
        output_frames[:] = frame_tensor
        output_tokens[:] = token_tensor
        output_scores.zero_()
        candidate_units = selected_units = candidate_tokens
    elif config.method == "block64_history":
        block_entries: list[tuple[int, int, int, int]] = []
        block_centroids = []
        for frame_index, frame in enumerate(frames):
            centroids = frame.block_centroids.to(query.device)
            block_centroids.append(centroids)
            for block_index, (start, end) in enumerate(
                zip(frame.block_starts.tolist(), frame.block_ends.tolist())
            ):
                block_entries.append((frame_index, block_index, int(start), int(end)))
        all_centroids = torch.cat(block_centroids, dim=2)
        scores = torch.einsum("bhqd,bhkd->bhqk", q_blocks.float(), all_centroids.float())
        scores = scores.amax(dim=2) / math.sqrt(dim)
        candidate_units = scores.shape[-1]

        for batch_index in range(batch):
            for head in range(heads):
                score_row = scores[batch_index, head].detach().cpu()
                order = torch.argsort(score_row, descending=True, stable=True)
                chosen_frames: list[int] = []
                chosen_tokens: list[int] = []
                chosen_scores: list[float] = []
                unit_count = 0
                for unit_index in order.tolist():
                    frame_index, _, start, end = block_entries[unit_index]
                    frame_id = frames[frame_index].frame_id
                    score = float(score_row[unit_index])
                    take = min(end - start, budget - len(chosen_tokens))
                    chosen_frames.extend([frame_id] * take)
                    chosen_tokens.extend(range(start, start + take))
                    chosen_scores.extend([score] * take)
                    unit_count += 1
                    if len(chosen_tokens) == budget:
                        break
                chosen_frames, chosen_tokens, chosen_scores = _sort_selected(
                    chosen_frames, chosen_tokens, chosen_scores
                )
                output_frames[batch_index, head] = torch.tensor(chosen_frames)
                output_tokens[batch_index, head] = torch.tensor(chosen_tokens)
                output_scores[batch_index, head] = torch.tensor(chosen_scores)
                selected_units += unit_count
    elif config.method == "kcluster32_history":
        cluster_entries: list[tuple[int, int]] = []
        centroids = []
        counts = []
        for frame_index, frame in enumerate(frames):
            centroids.append(frame.cluster_centroids.to(query.device))
            counts.append(frame.cluster_counts.to(query.device))
            cluster_entries.extend(
                (frame_index, cluster_index)
                for cluster_index in range(frame.cluster_centroids.shape[2])
            )
            current_min = int(frame.cluster_counts.min())
            current_max = int(frame.cluster_counts.max())
            cluster_min = current_min if cluster_min is None else min(cluster_min, current_min)
            cluster_max = current_max if cluster_max is None else max(cluster_max, current_max)
        all_centroids = torch.cat(centroids, dim=2)
        all_counts = torch.cat(counts, dim=2)
        scores = torch.einsum("bhqd,bhkd->bhqk", q_blocks.float(), all_centroids.float())
        scores = scores.amax(dim=2) / math.sqrt(dim) + all_counts.float().clamp_min(1).log()
        candidate_units = scores.shape[-1]

        for batch_index in range(batch):
            for head in range(heads):
                score_row = scores[batch_index, head].detach().cpu()
                order = torch.argsort(score_row, descending=True, stable=True)
                chosen_frames: list[int] = []
                chosen_tokens: list[int] = []
                chosen_scores: list[float] = []
                unit_count = 0
                for unit_index in order.tolist():
                    frame_index, cluster_index = cluster_entries[unit_index]
                    frame = frames[frame_index]
                    members = torch.nonzero(
                        frame.cluster_labels[batch_index, head] == cluster_index,
                        as_tuple=False,
                    ).flatten().tolist()
                    take = min(len(members), budget - len(chosen_tokens))
                    score = float(score_row[unit_index])
                    chosen_frames.extend([frame.frame_id] * take)
                    chosen_tokens.extend(members[:take])
                    chosen_scores.extend([score] * take)
                    unit_count += 1
                    if len(chosen_tokens) == budget:
                        break
                chosen_frames, chosen_tokens, chosen_scores = _sort_selected(
                    chosen_frames, chosen_tokens, chosen_scores
                )
                output_frames[batch_index, head] = torch.tensor(chosen_frames)
                output_tokens[batch_index, head] = torch.tensor(chosen_tokens)
                output_scores[batch_index, head] = torch.tensor(chosen_scores)
                selected_units += unit_count
    else:
        raise ValueError(f"unhandled sparse method: {config.method}")

    timing.routing_s = time.perf_counter() - route_start
    return SparseSelection(
        frame_ids=output_frames,
        token_ids=output_tokens,
        scores=output_scores,
        candidate_history_tokens=candidate_tokens,
        selected_history_tokens=budget,
        candidate_units=candidate_units,
        selected_units=selected_units,
        cluster_size_min=cluster_min,
        cluster_size_max=cluster_max,
        index_bytes=sum(frame.routing_bytes for frame in frames),
        timing=timing,
    )


def select_block64_from_tensor(
    query: torch.Tensor,
    key_unrotated: torch.Tensor,
    density: float,
    block_size: int,
) -> torch.Tensor:
    """Return sorted per-head token indices for native rolling-cache sparsity."""

    if query.ndim != 4 or key_unrotated.ndim != 4:
        raise ValueError("query/key must be [B,T,H,D]")
    batch, key_tokens, heads, dim = key_unrotated.shape
    budget = _exact_budget(key_tokens, density)
    q_blocks = _query_block_means(query, block_size)
    key_bhtd = key_unrotated.permute(0, 2, 1, 3)
    starts = list(range(0, key_tokens, block_size))
    centroids = torch.stack(
        [key_bhtd[:, :, start : start + block_size].float().mean(dim=2) for start in starts],
        dim=2,
    )
    scores = torch.einsum("bhqd,bhkd->bhqk", q_blocks.float(), centroids).amax(dim=2)
    scores = scores / math.sqrt(dim)
    selected = torch.empty((batch, heads, budget), dtype=torch.long, device=key_unrotated.device)
    for batch_index in range(batch):
        for head in range(heads):
            order = torch.argsort(scores[batch_index, head], descending=True, stable=True)
            indices: list[int] = []
            for block_index in order.tolist():
                start = starts[block_index]
                take = min(min(block_size, key_tokens - start), budget - len(indices))
                indices.extend(range(start, start + take))
                if len(indices) == budget:
                    break
            selected[batch_index, head] = torch.tensor(
                sorted(indices), dtype=torch.long, device=selected.device
            )
    return selected


def gather_per_head(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather ``[B,T,H,D]`` with different token indices for every head."""

    if tensor.ndim != 4 or indices.ndim != 3:
        raise ValueError("tensor must be [B,T,H,D] and indices [B,H,K]")
    batch, _, heads, dim = tensor.shape
    if indices.shape[:2] != (batch, heads):
        raise ValueError("batch/head mismatch for per-head gather")
    source = tensor.permute(0, 2, 1, 3)
    gathered = source.gather(2, indices.to(source.device).unsqueeze(-1).expand(-1, -1, -1, dim))
    return gathered.permute(0, 2, 1, 3).contiguous()
