"""Exact-budget history selectors for fixed blocks and variable clusters."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import SparseHistoryConfig
from .stats import TimingBreakdown


@dataclass
class FrameIndex:
    frame_id: int
    key: torch.Tensor
    value: torch.Tensor
    block_centroids: torch.Tensor
    block_starts: torch.Tensor
    block_ends: torch.Tensor
    cluster_centroids: torch.Tensor
    cluster_labels: torch.Tensor
    cluster_counts: torch.Tensor
    index_bytes: int
    routing_bytes: int
    archive_bytes: int
    index_elapsed_s: float


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
) -> FrameIndex:
    """Build both fixed-block and KCluster32 metadata for one archived frame."""

    if key_for_index.ndim != 4:
        raise ValueError("frame key must be [B,T,H,D]")
    if value_storage.shape != key_storage.shape or key_storage.shape != key_for_index.shape:
        raise ValueError("key/value storage and index tensors must share [B,T,H,D]")
    start_time = time.perf_counter()
    key_bhtd = key_for_index.permute(0, 2, 1, 3)
    _, _, tokens, _ = key_bhtd.shape

    if config.method == "block64_history":
        block_centroids = []
        starts = []
        ends = []
        for start in range(0, tokens, config.block_size):
            end = min(start + config.block_size, tokens)
            block_centroids.append(key_bhtd[:, :, start:end].float().mean(dim=2))
            starts.append(start)
            ends.append(end)
        block_centroids_tensor = torch.stack(block_centroids, dim=2)
        block_starts = torch.tensor(starts, dtype=torch.long, device=key_for_index.device)
        block_ends = torch.tensor(ends, dtype=torch.long, device=key_for_index.device)
    else:
        block_centroids_tensor = torch.empty(
            (*key_bhtd.shape[:2], 0, key_bhtd.shape[-1]),
            dtype=torch.float32,
            device=key_for_index.device,
        )
        block_starts = torch.empty(0, dtype=torch.long, device=key_for_index.device)
        block_ends = torch.empty(0, dtype=torch.long, device=key_for_index.device)

    if config.method == "kcluster32_history":
        cluster_centroids, cluster_labels, cluster_counts = _batched_spherical_kmeans(
            key_bhtd,
            config.clusters_per_frame,
            iterations=config.kmeans_iterations,
            tolerance=config.kmeans_tolerance,
            seed=config.seed + int(frame_id),
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
    index_elapsed = time.perf_counter() - start_time

    metadata = (
        block_centroids_tensor.detach().cpu(),
        block_starts.detach().cpu(),
        block_ends.detach().cpu(),
        cluster_centroids.detach().cpu(),
        cluster_labels.detach().cpu(),
        cluster_counts.detach().cpu(),
    )
    total_index_bytes = _tensor_bytes(*metadata)
    if config.method == "block64_history":
        routing_bytes = _tensor_bytes(metadata[0])
    elif config.method == "kcluster32_history":
        routing_bytes = _tensor_bytes(metadata[3], metadata[5])
    else:
        routing_bytes = 0
    return FrameIndex(
        frame_id=int(frame_id),
        key=key_storage,
        value=value_storage,
        block_centroids=metadata[0],
        block_starts=metadata[1],
        block_ends=metadata[2],
        cluster_centroids=metadata[3],
        cluster_labels=metadata[4],
        cluster_counts=metadata[5],
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
