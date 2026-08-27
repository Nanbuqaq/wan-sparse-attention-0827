"""Six distinct clean-room clustering route implementations."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _pad_features(vectors: torch.Tensor) -> torch.Tensor:
    width = vectors.shape[-1]
    target = max(16, 1 << (width - 1).bit_length())
    return F.pad(vectors, (0, target - width)) if target != width else vectors


def _timed_kmeans(
    vectors,
    *,
    config,
    state,
    layer,
    call_index,
    clusters,
    initial=None,
    cache_tag="",
):
    from adapters.routing import _timed_cuda, batched_euclidean_kmeans

    cache_key = f"{cache_tag}:{vectors.shape[-1]}:{clusters}"
    cached = state.cache().get(cache_key)
    first = cached is None and initial is None
    iterations = config.kmeans_init_iterations if first else config.kmeans_step_iterations
    result, elapsed = _timed_cuda(
        lambda: batched_euclidean_kmeans(
            vectors,
            clusters=clusters,
            iterations=iterations,
            seed=config.cluster_seed + layer * 1009 + call_index * 17,
            initial_centroids=initial if initial is not None else cached,
        ),
        enabled=config.measure_timing,
    )
    state.cache()[cache_key] = result.centroids.detach()
    return result, elapsed, iterations


def _finish(query, key, value, *, config, method, labels, cluster_ms, metadata):
    from adapters.routing import _fixed_plan

    return _fixed_plan(
        query,
        key,
        value,
        config=config,
        method=method,
        k_labels=labels,
        cluster_ms=cluster_ms,
        metadata=metadata,
    )


def route_capacity_balanced(query, key, value, *, config, state, layer, call_index):
    clusters = int(config.route_params.get("clusters", 128))
    capacity_factor = float(config.route_params.get("capacity_factor", 1.5))
    result, cluster_ms, iterations = _timed_kmeans(
        key,
        config=config,
        state=state,
        layer=layer,
        call_index=call_index,
        clusters=clusters,
    )
    batch, heads, length = result.labels.shape
    capacity = max(1, int(math.ceil(length / clusters * capacity_factor)))
    flat = result.labels.reshape(batch * heads, length)
    order = torch.argsort(flat, dim=1, stable=True)
    sorted_labels = flat.gather(1, order)
    counts = torch.stack(
        [torch.bincount(row, minlength=clusters) for row in flat], dim=0
    )
    starts = torch.cumsum(counts, dim=1) - counts
    position = torch.arange(length, device=key.device).view(1, length)
    within = position - starts.gather(1, sorted_labels)
    split = torch.div(within, capacity, rounding_mode="floor")
    max_pieces = max(1, int(math.ceil(length / clusters / capacity)) + 2)
    sorted_new = sorted_labels * max_pieces + split
    new_labels = torch.empty_like(flat)
    new_labels.scatter_(1, order, sorted_new)
    labels = new_labels.view(batch, heads, length)
    return _finish(
        query,
        key,
        value,
        config=config,
        method="capacity_balanced",
        labels=labels,
        cluster_ms=cluster_ms,
        metadata={
            "family": "capacity_constrained_kmeans",
            "base_clusters": clusters,
            "capacity": capacity,
            "capacity_factor": capacity_factor,
            "iterations": iterations,
            "effective_clusters_mean": float(
                torch.tensor(
                    [torch.unique(row).numel() for row in new_labels],
                    device=key.device,
                    dtype=torch.float32,
                ).mean()
            ),
        },
    )


def route_radius_adaptive(query, key, value, *, config, state, layer, call_index):
    base_clusters = int(config.route_params.get("base_clusters", 64))
    max_added = int(config.route_params.get("max_added_clusters", 64))
    threshold = float(config.route_params.get("radius_threshold", 4.0))
    base, first_ms, iterations = _timed_kmeans(
        key,
        config=config,
        state=state,
        layer=layer,
        call_index=call_index,
        clusters=base_clusters,
    )
    assigned = base.centroids.gather(
        2,
        base.labels.unsqueeze(-1).expand(*base.labels.shape, base.centroids.shape[-1]),
    )
    residual = (key.float() - assigned).norm(dim=-1)
    outlier_fraction = float((residual > threshold).float().mean())
    added = max(1, min(max_added, int(round(max_added * max(outlier_fraction, 0.05)))))
    top = torch.topk(residual, k=added, dim=2).indices
    seeds = key.float().gather(
        2, top.unsqueeze(-1).expand(*top.shape, key.shape[-1])
    )
    initial = torch.cat((base.centroids.float(), seeds), dim=2)
    adaptive, second_ms, second_iterations = _timed_kmeans(
        key,
        config=config,
        state=state,
        layer=layer,
        call_index=call_index + 1,
        clusters=base_clusters + added,
        initial=initial,
    )
    return _finish(
        query,
        key,
        value,
        config=config,
        method="radius_adaptive",
        labels=adaptive.labels,
        cluster_ms=first_ms + second_ms,
        metadata={
            "family": "dpmeans_residual_seeded",
            "base_clusters": base_clusters,
            "added_clusters": added,
            "radius_threshold": threshold,
            "outlier_fraction": outlier_fraction,
            "iterations": [iterations, second_iterations],
        },
    )


def route_hierarchical(query, key, value, *, config, state, layer, call_index):
    coarse_clusters = int(config.route_params.get("coarse_clusters", 32))
    branches = int(config.route_params.get("branches", 4))
    coarse, first_ms, iterations = _timed_kmeans(
        key,
        config=config,
        state=state,
        layer=layer,
        call_index=call_index,
        clusters=coarse_clusters,
    )
    batch, heads, length, dim = key.shape
    seeds = []
    for branch in range(branches):
        score = (key.float() * coarse.centroids.gather(
            2,
            coarse.labels.unsqueeze(-1).expand(batch, heads, length, dim),
        )).sum(dim=-1)
        score = score + branch * 1e-6 * torch.arange(length, device=key.device)
        branch_ids = []
        for cluster in range(coarse_clusters):
            masked = score.masked_fill(coarse.labels != cluster, -float("inf"))
            rank = min(branch, length - 1)
            branch_ids.append(torch.topk(masked, k=rank + 1, dim=2).indices[..., -1])
        ids = torch.stack(branch_ids, dim=2)
        seeds.append(key.float().gather(2, ids.unsqueeze(-1).expand(*ids.shape, dim)))
    initial = torch.stack(seeds, dim=3).reshape(batch, heads, coarse_clusters * branches, dim)
    fine, second_ms, fine_iterations = _timed_kmeans(
        key,
        config=config,
        state=state,
        layer=layer,
        call_index=call_index + 1,
        clusters=coarse_clusters * branches,
        initial=initial,
    )
    return _finish(
        query,
        key,
        value,
        config=config,
        method="hierarchical",
        labels=fine.labels,
        cluster_ms=first_ms + second_ms,
        metadata={
            "family": "coarse_to_fine_kmeans",
            "coarse_clusters": coarse_clusters,
            "branches": branches,
            "iterations": [iterations, fine_iterations],
        },
    )


def route_product_quantized(query, key, value, *, config, state, layer, call_index):
    subspaces = int(config.route_params.get("subspaces", 4))
    codebook = int(config.route_params.get("codebook_clusters", 16))
    if key.shape[-1] % subspaces:
        raise ValueError("head dimension must be divisible by PQ subspaces")
    width = key.shape[-1] // subspaces
    codes = torch.zeros(key.shape[:3], dtype=torch.long, device=key.device)
    total_ms = 0.0
    iteration_values = []
    for index in range(subspaces):
        result, elapsed, iterations = _timed_kmeans(
            key[..., index * width : (index + 1) * width],
            config=config,
            state=state,
            layer=layer,
            call_index=call_index + index,
            clusters=codebook,
            cache_tag=f"pq_subspace_{index}",
        )
        codes += result.labels * (codebook**index)
        total_ms += elapsed
        iteration_values.append(iterations)
    return _finish(
        query,
        key,
        value,
        config=config,
        method="product_quantized",
        labels=codes,
        cluster_ms=total_ms,
        metadata={
            "family": "product_quantization",
            "subspaces": subspaces,
            "codebook_clusters": codebook,
            "observed_codes_mean": float(
                torch.tensor(
                    [torch.unique(row).numel() for row in codes.flatten(0, 1)],
                    device=key.device,
                    dtype=torch.float32,
                ).mean()
            ),
            "iterations": iteration_values,
        },
    )


def route_spatiotemporal(query, key, value, *, config, state, layer, call_index):
    clusters = int(config.route_params.get("clusters", 128))
    weight = float(config.route_params.get("position_weight", 0.25))
    content_dims = int(config.route_params.get("content_dims", 64))
    frames = int(config.route_params.get("frames_latent", 21))
    height = int(config.route_params.get("height_latent", 30))
    width = int(config.route_params.get("width_latent", 52))
    length = key.shape[2]
    if frames * height * width != length:
        raise ValueError("spatiotemporal layout mismatch")
    token = torch.arange(length, device=key.device)
    t = token // (height * width)
    rem = token % (height * width)
    y = rem // width
    x = rem % width
    position = torch.stack(
        (
            t.float() / max(1, frames - 1),
            y.float() / max(1, height - 1),
            x.float() / max(1, width - 1),
        ),
        dim=1,
    )
    position = position.view(1, 1, length, 3).expand(key.shape[0], key.shape[1], -1, -1)
    augmented = _pad_features(
        torch.cat(
            (
                F.normalize(key.float(), dim=-1)[..., :content_dims],
                position * weight,
            ),
            dim=-1,
        )
    )
    result, cluster_ms, iterations = _timed_kmeans(
        augmented,
        config=config,
        state=state,
        layer=layer,
        call_index=call_index,
        clusters=clusters,
    )
    return _finish(
        query,
        key,
        value,
        config=config,
        method="spatiotemporal",
        labels=result.labels,
        cluster_ms=cluster_ms,
        metadata={
            "family": "feature_augmented_spatiotemporal_kmeans",
            "clusters": clusters,
            "position_weight": weight,
            "content_dims": content_dims,
            "iterations": iterations,
        },
    )


def route_query_metric(query, key, value, *, config, state, layer, call_index):
    clusters = int(config.route_params.get("clusters", 128))
    rank = int(config.route_params.get("rank", 32))
    refresh = int(config.route_params.get("basis_refresh_calls", 20))
    basis_key = f"query_metric_basis:{layer}"
    cache = state.cache()
    basis = cache.get(basis_key)
    basis_age = int(cache.get(basis_key + ":age", refresh))
    if basis is None or basis_age >= refresh:
        sample_count = min(2048, query.shape[2])
        ids = torch.linspace(0, query.shape[2] - 1, sample_count, device=query.device).round().long()
        sample = F.normalize(query.index_select(2, ids).float(), dim=-1)
        centered = sample - sample.mean(dim=2, keepdim=True)
        covariance = torch.matmul(centered.transpose(-2, -1), centered) / max(1, sample_count - 1)
        _, eigenvectors = torch.linalg.eigh(covariance)
        basis = eigenvectors[..., -min(rank, query.shape[-1]) :].contiguous()
        cache[basis_key] = basis.detach()
        basis_age = 0
    cache[basis_key + ":age"] = basis_age + 1
    projected = _pad_features(torch.matmul(key.float(), basis))
    result, cluster_ms, iterations = _timed_kmeans(
        projected,
        config=config,
        state=state,
        layer=layer,
        call_index=call_index,
        clusters=clusters,
    )
    return _finish(
        query,
        key,
        value,
        config=config,
        method="query_metric",
        labels=result.labels,
        cluster_ms=cluster_ms,
        metadata={
            "family": "query_covariance_metric_kmeans",
            "clusters": clusters,
            "rank": basis.shape[-1],
            "basis_refresh_calls": refresh,
            "iterations": iterations,
        },
    )
