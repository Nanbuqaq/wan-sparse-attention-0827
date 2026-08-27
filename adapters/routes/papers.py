"""Clean-room/pinned-upstream paper-method route implementations."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _pad_features(vectors: torch.Tensor) -> torch.Tensor:
    width = vectors.shape[-1]
    target = max(16, 1 << (width - 1).bit_length())
    return F.pad(vectors, (0, target - width)) if target != width else vectors


def _legacy(*args, **kwargs):
    from adapters.routing import _route_attention_legacy

    return _route_attention_legacy(*args, **kwargs)


def route_svg2(*args, **kwargs):
    return _legacy(*args, **kwargs)


def route_adacluster(query, key, value, *, config, state, layer, call_index):
    from adapters.routing import (
        KMeansResult,
        _cluster_route_plan,
        _timed_cuda,
        batched_euclidean_kmeans,
    )

    q_clusters = int(config.route_params.get("q_clusters", config.q_clusters))
    initial_k = int(config.route_params.get("initial_k_clusters", 100))
    max_added = int(config.route_params.get("max_added_clusters", 64))
    threshold = float(config.route_params.get("distance_threshold", 5.5))
    reuse_calls = int(config.route_params.get("reuse_calls", 20))
    cache = state.cache()
    refresh = call_index % max(1, reuse_calls) == 0 or "ada_results" not in cache
    first = cache.get("ada_q_centroids") is None
    iterations = config.kmeans_init_iterations if first else config.kmeans_step_iterations

    def cluster():
        normalized_q = F.normalize(query.float(), dim=-1, eps=1e-12)
        q_result = batched_euclidean_kmeans(
            normalized_q,
            clusters=q_clusters,
            iterations=iterations,
            seed=config.cluster_seed + layer * 1009,
            initial_centroids=cache.get("ada_q_centroids"),
        )
        base = batched_euclidean_kmeans(
            key,
            clusters=initial_k,
            iterations=iterations,
            seed=config.cluster_seed + 97 + layer * 1009,
            initial_centroids=cache.get("ada_k_base_centroids"),
        )
        assigned = base.centroids.gather(
            2,
            base.labels.unsqueeze(-1).expand(*base.labels.shape, key.shape[-1]),
        )
        residual = (key.float() - assigned).norm(dim=-1)
        fraction = float((residual > threshold).float().mean())
        added = max(1, min(max_added, int(round(max_added * max(fraction, 0.05)))))
        ids = torch.topk(residual, k=added, dim=2).indices
        seeds = key.float().gather(2, ids.unsqueeze(-1).expand(*ids.shape, key.shape[-1]))
        initial = torch.cat((base.centroids, seeds), dim=2)
        k_result = batched_euclidean_kmeans(
            key,
            clusters=initial_k + added,
            iterations=iterations,
            seed=config.cluster_seed + 193 + layer * 1009,
            initial_centroids=initial,
        )
        flat_key = key.float().flatten(0, 1)
        flat_labels = k_result.labels.flatten(0, 1)
        groups = k_result.centroids.shape[2]
        offsets = torch.arange(flat_labels.shape[0], device=key.device).view(-1, 1) * groups
        global_labels = flat_labels + offsets
        index = global_labels.reshape(-1, 1).expand(-1, key.shape[-1])
        minimum = torch.full(
            (flat_labels.shape[0] * groups, key.shape[-1]),
            float("inf"),
            device=key.device,
        )
        maximum = torch.full_like(minimum, -float("inf"))
        minimum.scatter_reduce_(0, index, flat_key.reshape(-1, key.shape[-1]), reduce="amin", include_self=True)
        maximum.scatter_reduce_(0, index, flat_key.reshape(-1, key.shape[-1]), reduce="amax", include_self=True)
        minimum = minimum.view(*k_result.centroids.shape)
        maximum = maximum.view(*k_result.centroids.shape)
        return q_result, k_result, base.centroids, minimum, maximum, fraction, added

    if refresh:
        (
            q_result,
            k_result,
            base_centroids,
            minimum,
            maximum,
            outlier_fraction,
            added,
        ), cluster_ms = _timed_cuda(cluster, enabled=config.measure_timing)
        cache["ada_q_centroids"] = q_result.centroids.detach()
        cache["ada_k_base_centroids"] = base_centroids.detach()
        cache["ada_k_centroids"] = k_result.centroids.detach()
        cache["ada_results"] = (
            q_result,
            k_result,
            minimum,
            maximum,
            outlier_fraction,
            added,
        )
    else:
        q_result, k_result, minimum, maximum, outlier_fraction, added = cache["ada_results"]
        cluster_ms = 0.0
    q_centroids = F.normalize(q_result.centroids.float(), dim=-1, eps=1e-12)
    scores = torch.matmul(q_centroids.clamp_min(0), maximum.transpose(-2, -1))
    scores += torch.matmul(q_centroids.clamp_max(0), minimum.transpose(-2, -1))
    return _cluster_route_plan(
        query,
        key,
        value,
        config=config,
        method="adacluster",
        q_result=q_result,
        k_result=k_result,
        cluster_scores=scores,
        cluster_ms=cluster_ms,
        metadata={
            "provenance": "clean_room_from_paper_and_public_interface",
            "q_clusters": q_clusters,
            "initial_k_clusters": initial_k,
            "added_clusters": added,
            "distance_threshold": threshold,
            "outlier_fraction": outlier_fraction,
            "iterations": iterations,
            "reuse_calls": reuse_calls,
            "refreshed": refresh,
            "selector": "sign_aware_cluster_upper_bound",
        },
    )


def route_svoo(query, key, value, *, config, state, layer, call_index):
    from adapters.routing import KMeansResult, _cluster_route_plan, _core, _timed_cuda

    q_clusters = int(config.route_params.get("q_clusters", config.q_clusters))
    k_clusters = int(config.route_params.get("k_clusters", config.k_clusters))
    iterations = int(config.route_params.get("co_cluster_iterations", 2))
    reuse_calls = int(config.route_params.get("reuse_calls", 1))
    cache = state.cache()
    refresh = call_index % max(1, reuse_calls) == 0 or "svoo_results" not in cache

    if refresh:
        def cluster():
            torch.manual_seed(config.cluster_seed + layer * 1009 + call_index * 17)
            qlabels, qcentroids, qsizes, klabels, kcentroids, ksizes = _core().co_cluster_tokens(
                query.float().flatten(0, 1).contiguous(),
                key.float().flatten(0, 1).contiguous(),
                q_clusters,
                k_clusters,
                max_iters=iterations,
            )
            batch, heads, length, dim = query.shape
            q_result = KMeansResult(
                labels=qlabels.view(batch, heads, length),
                centroids=qcentroids.view(batch, heads, q_clusters, dim),
                sizes=qsizes.view(batch, heads, q_clusters).to(torch.int32),
                empty_clusters=int((qsizes == 0).sum()),
            )
            k_result = KMeansResult(
                labels=klabels.view(batch, heads, length),
                centroids=kcentroids.view(batch, heads, k_clusters, dim),
                sizes=ksizes.view(batch, heads, k_clusters).to(torch.int32),
                empty_clusters=int((ksizes == 0).sum()),
            )
            return q_result, k_result

        (q_result, k_result), cluster_ms = _timed_cuda(cluster, enabled=config.measure_timing)
        cache["svoo_results"] = (q_result, k_result)
    else:
        q_result, k_result = cache["svoo_results"]
        cluster_ms = 0.0
    scores = torch.matmul(
        q_result.centroids.float(), k_result.centroids.float().transpose(-2, -1)
    ) / math.sqrt(query.shape[-1])
    scores += torch.log(k_result.sizes.unsqueeze(-2).float().clamp_min(1))
    return _cluster_route_plan(
        query,
        key,
        value,
        config=config,
        method="svoo",
        q_result=q_result,
        k_result=k_result,
        cluster_scores=scores,
        cluster_ms=cluster_ms,
        metadata={
            "provenance": "SVOO_Apache_2_0_core",
            "q_clusters": q_clusters,
            "k_clusters": k_clusters,
            "co_cluster_iterations": iterations,
            "reuse_calls": reuse_calls,
            "refreshed": refresh,
        },
    )


def route_scope(query, key, value, *, config, state, layer, call_index):
    from adapters.routing import _fixed_plan, _timed_cuda, batched_euclidean_kmeans, block_means, fixed_block_sizes

    if config.backend != "fixed64_bf16":
        raise ValueError("paper-derived SCOPE currently uses the validated fixed64 backend")
    q_clusters = int(config.route_params.get("q_clusters", config.q_clusters))
    subspace_clusters = int(config.route_params.get("subspace_clusters", config.k_clusters))
    iterations = config.kmeans_init_iterations if state.cache().get("scope_q") is None else config.kmeans_step_iterations
    dimensions = query.shape[-1]
    height_dim = width_dim = 2 * (dimensions // 6)
    temporal_dim = dimensions - height_dim - width_dim
    slices = (
        slice(0, temporal_dim),
        slice(temporal_dim, temporal_dim + height_dim),
        slice(temporal_dim + height_dim, dimensions),
    )

    def cluster_and_score():
        q_result = batched_euclidean_kmeans(
            F.normalize(query.float(), dim=-1),
            clusters=q_clusters,
            iterations=iterations,
            seed=config.cluster_seed + layer * 1009,
            initial_centroids=state.cache().get("scope_q"),
        )
        proxy = torch.zeros(
            (*q_result.centroids.shape[:3], key.shape[2]),
            device=key.device,
            dtype=torch.float32,
        )
        subspace_meta = []
        for index, feature_slice in enumerate(slices):
            result = batched_euclidean_kmeans(
                _pad_features(key[..., feature_slice]),
                clusters=subspace_clusters,
                iterations=iterations,
                seed=config.cluster_seed + 97 + index + layer * 1009,
                initial_centroids=state.cache().get(f"scope_k_{index}"),
            )
            table = torch.matmul(
                _pad_features(q_result.centroids[..., feature_slice].float()),
                result.centroids.float().transpose(-2, -1),
            )
            proxy += table.gather(
                3,
                result.labels.unsqueeze(2).expand(-1, -1, q_clusters, -1),
            )
            state.cache()[f"scope_k_{index}"] = result.centroids.detach()
            subspace_meta.append(int((result.sizes == 0).sum()))
        state.cache()["scope_q"] = q_result.centroids.detach()
        return q_result, proxy / math.sqrt(dimensions), subspace_meta

    (q_result, proxy, empty_counts), cluster_ms = _timed_cuda(
        cluster_and_score, enabled=config.measure_timing
    )
    batch, heads, length, _ = query.shape
    sizes = fixed_block_sizes(batch, heads, length, config.block_size, query.device)
    q_means = block_means(query, sizes, config.block_size)
    q_group = torch.matmul(
        q_means.float(), q_result.centroids.float().transpose(-2, -1)
    ).argmax(dim=-1)
    blocks = sizes.shape[-1]
    padded_proxy = F.pad(proxy, (0, blocks * config.block_size - length))
    proxy_blocks = padded_proxy.view(batch, heads, q_clusters, blocks, config.block_size).sum(dim=-1)
    proxy_blocks = proxy_blocks / sizes.unsqueeze(2).float()
    scores = proxy_blocks.gather(
        2,
        q_group.unsqueeze(-1).expand(batch, heads, blocks, blocks),
    )
    return _fixed_plan(
        query,
        key,
        value,
        config=config,
        method="scope",
        score_override=scores,
        cluster_ms=cluster_ms,
        metadata={
            "provenance": "paper_derived_no_official_code",
            "q_clusters": q_clusters,
            "subspace_clusters": subspace_clusters,
            "subspace_dims": [temporal_dim, height_dim, width_dim],
            "empty_subspace_clusters": empty_counts,
            "iterations": iterations,
        },
    )
