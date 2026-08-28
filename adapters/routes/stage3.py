"""Stage-3 stable-coverage, V-aware, and final hybrid routes.

All routes preserve the original token order.  A fixed part of the exact edge
budget retains Original-Block connections, a second part guarantees local/time
coverage, and clustering is only allowed to spend the remaining remote budget.
The executed Attention still consumes the original Q/K/V tensors.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _standardize(scores: torch.Tensor) -> torch.Tensor:
    mean = scores.mean(dim=-1, keepdim=True)
    scale = scores.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (scores - mean) / scale


def _block_geometry(length: int, block_size: int, route_params: dict, device):
    blocks = math.ceil(length / block_size)
    frames = int(route_params.get("frames_latent", 21))
    height = int(route_params.get("height_latent", 30))
    width = int(route_params.get("width_latent", 52))
    if frames * height * width != length:
        raise ValueError(
            f"Stage-3 layout {frames}x{height}x{width} does not match length {length}"
        )
    centers = (
        torch.arange(blocks, device=device) * block_size + block_size // 2
    ).clamp_max(length - 1)
    time_ids = centers // (height * width)
    remainder = centers % (height * width)
    y_ids = remainder // width
    x_ids = remainder % width
    coordinates = torch.stack(
        (
            time_ids.float() / max(1, frames - 1),
            y_ids.float() / max(1, height - 1),
            x_ids.float() / max(1, width - 1),
        ),
        dim=-1,
    )
    weights = torch.tensor(
        route_params.get("local_distance_weights", [2.0, 1.0, 1.0]),
        device=device,
        dtype=torch.float32,
    )
    local_score = -(
        (coordinates[:, None] - coordinates[None, :]).abs() * weights
    ).sum(dim=-1)
    remote_min_frames = int(route_params.get("remote_min_frames", 2))
    remote_mask = (time_ids[:, None] - time_ids[None, :]).abs() >= remote_min_frames
    return local_score, remote_mask, time_ids


def _tiered_priority(
    direct_scores: torch.Tensor,
    local_scores: torch.Tensor,
    remote_scores: torch.Tensor,
    remote_mask: torch.Tensor,
    *,
    density: float,
    base_fraction: float,
    local_fraction: float,
) -> tuple[torch.Tensor, dict]:
    """Encode an exact per-row three-tier allocation for fixed_edge_budget_map."""
    if base_fraction < 0 or local_fraction < 0 or base_fraction + local_fraction > 1:
        raise ValueError("invalid Stage-3 budget fractions")
    batch, heads, q_blocks, k_blocks = direct_scores.shape
    row_budget = max(1, min(k_blocks, int(round(k_blocks * density))))
    base_count = max(1, min(row_budget, int(round(row_budget * base_fraction))))
    local_count = max(
        0,
        min(row_budget - base_count, int(round(row_budget * local_fraction))),
    )
    remote_count = row_budget - base_count - local_count

    base = torch.zeros_like(direct_scores, dtype=torch.bool)
    base.scatter_(-1, torch.topk(direct_scores, k=base_count, dim=-1).indices, True)

    local = torch.zeros_like(base)
    if local_count:
        local_candidates = local_scores.view(1, 1, q_blocks, k_blocks).expand_as(
            direct_scores
        )
        local_candidates = local_candidates.masked_fill(base, -float("inf"))
        local.scatter_(
            -1,
            torch.topk(local_candidates, k=local_count, dim=-1).indices,
            True,
        )

    remote = torch.zeros_like(base)
    if remote_count:
        unavailable = base | local | ~remote_mask.view(1, 1, q_blocks, k_blocks)
        candidates = remote_scores.masked_fill(unavailable, -float("inf"))
        finite = torch.isfinite(candidates).sum(dim=-1)
        if int(finite.min()) < remote_count:
            candidates = remote_scores.masked_fill(base | local, -float("inf"))
        remote.scatter_(
            -1,
            torch.topk(candidates, k=remote_count, dim=-1).indices,
            True,
        )

    # Fixed-edge selection sees exactly row_budget high-priority entries.  The
    # low-amplitude standardized terms make ties deterministic and auditable.
    priority = _standardize(direct_scores) * 1e-3
    priority = priority + base.float() * 3e6
    priority = priority + local.float() * 2e6
    priority = priority + remote.float() * 1e6
    return priority, {
        "row_budget": row_budget,
        "base_edges_per_row": base_count,
        "local_edges_per_row": local_count,
        "remote_edges_per_row": remote_count,
        "base_fraction_executed": base_count / row_budget,
        "local_fraction_executed": local_count / row_budget,
        "remote_fraction_executed": remote_count / row_budget,
    }


def _cluster_block_proxy(
    key: torch.Tensor,
    q_means: torch.Tensor,
    *,
    config,
    state,
    layer: int,
    call_index: int,
):
    from adapters.routing import _timed_cuda, batched_euclidean_kmeans

    clusters = int(config.route_params.get("remote_clusters", 128))
    refresh_calls = int(config.route_params.get("refresh_calls", 10))
    cache = state.cache()
    cache_key = f"stage3_remote:{clusters}"
    refresh = cache_key not in cache or call_index % max(1, refresh_calls) == 0

    if refresh:
        iterations = (
            config.kmeans_init_iterations
            if cache.get(cache_key + ":centroids") is None
            else config.kmeans_step_iterations
        )

        def cluster():
            return batched_euclidean_kmeans(
                key,
                clusters=clusters,
                iterations=iterations,
                seed=config.cluster_seed + layer * 1009 + call_index * 17,
                initial_centroids=cache.get(cache_key + ":centroids"),
            )

        result, cluster_ms = _timed_cuda(cluster, enabled=config.measure_timing)
        batch, heads, length = result.labels.shape
        blocks = math.ceil(length / config.block_size)

        def build_membership():
            padded_length = blocks * config.block_size
            labels = F.pad(result.labels, (0, padded_length - length), value=clusters)
            membership = F.one_hot(labels, num_classes=clusters + 1)[..., :clusters]
            membership = membership.view(
                batch, heads, blocks, config.block_size, clusters
            ).float().sum(dim=3)
            valid = torch.full(
                (blocks,), config.block_size, device=key.device, dtype=torch.float32
            )
            if length % config.block_size:
                valid[-1] = length % config.block_size
            return membership / valid.view(1, 1, blocks, 1)

        membership, membership_ms = _timed_cuda(
            build_membership, enabled=config.measure_timing
        )
        cluster_ms += membership_ms
        cache[cache_key] = (result.centroids.detach(), membership.detach())
        cache[cache_key + ":centroids"] = result.centroids.detach()
        empty_clusters = result.empty_clusters
    else:
        centroids, membership = cache[cache_key]
        cluster_ms = 0.0
        iterations = 0
        empty_clusters = int((membership.sum(dim=2) == 0).sum())

    centroids, membership = cache[cache_key]

    def score_proxy():
        q_to_cluster = torch.matmul(
            q_means.float(), centroids.float().transpose(-2, -1)
        ) / math.sqrt(key.shape[-1])
        return torch.matmul(q_to_cluster, membership.transpose(-2, -1))

    proxy, proxy_ms = _timed_cuda(score_proxy, enabled=config.measure_timing)
    return proxy, cluster_ms, {
        "remote_clusters": clusters,
        "refresh_calls": refresh_calls,
        "refreshed": refresh,
        "cluster_iterations": iterations,
        "empty_clusters": empty_clusters,
        "cluster_proxy_ms": proxy_ms,
    }


def _value_remote_score(
    direct_scores: torch.Tensor,
    v_means: torch.Tensor,
    stable_map: torch.Tensor,
    *,
    objective: str,
) -> torch.Tensor:
    probability = torch.softmax(direct_scores.float(), dim=-1)
    v_norm = torch.linalg.vector_norm(v_means.float(), dim=-1).clamp_min(1e-6)
    if objective == "v_norm":
        return v_norm.unsqueeze(-2).expand_as(probability)
    if objective == "v_prototype":
        return probability * v_norm.unsqueeze(-2)
    if objective != "output_residual":
        raise ValueError(f"unknown V-aware objective: {objective}")
    stable_probability = probability * stable_map.float()
    stable_probability = stable_probability / stable_probability.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)
    stable_output = torch.matmul(stable_probability, v_means.float())
    output_sq = (stable_output * stable_output).sum(dim=-1, keepdim=True)
    value_sq = (v_means.float() * v_means.float()).sum(dim=-1).unsqueeze(-2)
    cross = torch.matmul(stable_output, v_means.float().transpose(-2, -1))
    residual = (output_sq + value_sq - 2 * cross).clamp_min(0).sqrt()
    return probability * residual


def _route_hybrid(
    query,
    key,
    value,
    *,
    config,
    state,
    layer,
    call_index,
    method: str,
    v_aware: bool,
    adaptive_schedule: bool,
):
    from adapters.routing import (
        _fixed_plan,
        _timed_cuda,
        block_means,
        fixed_block_sizes,
    )

    batch, heads, length, dim = query.shape
    def prepare_blocks():
        sizes = fixed_block_sizes(batch, heads, length, config.block_size, query.device)
        q_means = block_means(query, sizes, config.block_size)
        k_means = block_means(key, sizes, config.block_size)
        v_means = block_means(value, sizes, config.block_size)
        direct_scores = torch.matmul(
            q_means.float(), k_means.float().transpose(-2, -1)
        ) / math.sqrt(dim)
        return sizes, q_means, v_means, direct_scores

    (sizes, q_means, v_means, direct_scores), block_prepare_ms = _timed_cuda(
        prepare_blocks, enabled=config.measure_timing
    )
    cluster_proxy, cluster_ms, cluster_meta = _cluster_block_proxy(
        key,
        q_means,
        config=config,
        state=state,
        layer=layer,
        call_index=call_index,
    )
    local_scores, remote_mask, time_ids = _block_geometry(
        length, config.block_size, config.route_params, query.device
    )
    base_fraction = float(config.route_params.get("base_fraction", 0.80))
    local_fraction = float(config.route_params.get("local_fraction", 0.10))
    phase = call_index / max(1, config.inference_steps * config.calls_per_step - 1)
    if adaptive_schedule:
        early_bonus = float(config.route_params.get("early_base_bonus", 0.05))
        late_bonus = float(config.route_params.get("late_base_bonus", 0.025))
        if phase < 0.20:
            base_fraction += early_bonus
        elif phase > 0.80:
            base_fraction += late_bonus
        base_fraction = min(base_fraction, 1.0 - local_fraction)

    def plan_scores():
        # First derive the stable map.  V-aware scoring is only allowed to rank
        # the remote remainder and can never evict the Block/local guarantees.
        initial_priority, initial_meta = _tiered_priority(
            direct_scores,
            local_scores,
            cluster_proxy,
            remote_mask,
            density=config.density,
            base_fraction=base_fraction,
            local_fraction=local_fraction,
        )
        if not v_aware:
            return initial_priority, initial_meta, "qk_cluster"
        stable_map = initial_priority >= 1.5e6
        objective = str(config.route_params.get("v_objective", "output_residual"))
        value_score = _value_remote_score(
            direct_scores, v_means, stable_map, objective=objective
        )
        weight = float(config.route_params.get("v_weight", 0.75))
        remote_score = _standardize(cluster_proxy) + weight * _standardize(value_score)
        priority, allocation = _tiered_priority(
            direct_scores,
            local_scores,
            remote_score,
            remote_mask,
            density=config.density,
            base_fraction=base_fraction,
            local_fraction=local_fraction,
        )
        return priority, allocation, objective

    (priority, allocation, objective), planner_ms = _timed_cuda(
        plan_scores, enabled=config.measure_timing
    )
    metadata = {
        "family": "stable_block_local_plus_remote_cluster",
        "preserves_original_token_order": True,
        "executes_original_qkv": True,
        "remote_objective": objective,
        "v_aware": v_aware,
        "adaptive_schedule": adaptive_schedule,
        "denoise_phase": phase,
        "base_fraction_requested": base_fraction,
        "local_fraction_requested": local_fraction,
        "remote_time_min": int(config.route_params.get("remote_min_frames", 2)),
        "time_block_min": int(time_ids.min()),
        "time_block_max": int(time_ids.max()),
        "planner_score_ms": planner_ms,
        "block_prepare_ms": block_prepare_ms,
        **cluster_meta,
        **allocation,
    }
    return _fixed_plan(
        query,
        key,
        value,
        config=config,
        method=method,
        score_override=priority,
        cluster_ms=cluster_ms,
        preselection_ms=block_prepare_ms + planner_ms + float(cluster_meta["cluster_proxy_ms"]),
        metadata=metadata,
    )


def route_coverage_cluster(query, key, value, *, config, state, layer, call_index):
    return _route_hybrid(
        query,
        key,
        value,
        config=config,
        state=state,
        layer=layer,
        call_index=call_index,
        method="coverage_cluster",
        v_aware=False,
        adaptive_schedule=False,
    )


def route_vaware_cluster(query, key, value, *, config, state, layer, call_index):
    return _route_hybrid(
        query,
        key,
        value,
        config=config,
        state=state,
        layer=layer,
        call_index=call_index,
        method="vaware_cluster",
        v_aware=True,
        adaptive_schedule=False,
    )


def route_stage3_hybrid(query, key, value, *, config, state, layer, call_index):
    return _route_hybrid(
        query,
        key,
        value,
        config=config,
        state=state,
        layer=layer,
        call_index=call_index,
        method="stage3_hybrid",
        v_aware=True,
        adaptive_schedule=True,
    )
