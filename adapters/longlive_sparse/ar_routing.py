"""Backend-independent rectangular Q-to-history routing for LongLive AR methods."""

from __future__ import annotations

import math
from dataclasses import replace

import torch
import torch.nn.functional as F

from .methods import MethodSpec, method_spec
from .route_plan import HistoryRoutePlan


def _kmeans(
    vectors: torch.Tensor,
    clusters: int,
    *,
    iterations: int,
    seed: int,
    spherical: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if vectors.ndim != 2:
        raise ValueError("kmeans vectors must be [tokens,dim]")
    tokens, dim = vectors.shape
    clusters = max(1, min(int(clusters), tokens))
    work = vectors.float()
    if spherical:
        work = F.normalize(work, dim=-1, eps=1e-12)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    initial = torch.randperm(tokens, generator=generator)[:clusters].to(work.device)
    centroids = work.index_select(0, initial).clone()
    labels = torch.full((tokens,), -1, dtype=torch.long, device=work.device)
    for _ in range(max(1, iterations)):
        previous = labels.clone()
        if spherical:
            scores = work @ centroids.T
            best, labels = scores.max(dim=1)
        else:
            distances = torch.cdist(work, centroids)
            best, labels = distances.min(dim=1)
        counts = torch.bincount(labels, minlength=clusters)
        empty = torch.nonzero(counts == 0, as_tuple=False).flatten()
        if empty.numel():
            candidates = torch.argsort(best, descending=spherical, stable=True)
            for cluster in empty.tolist():
                for token in candidates.tolist():
                    old = int(labels[token])
                    if int(counts[old]) > 1:
                        labels[token] = cluster
                        counts[old] -= 1
                        counts[cluster] = 1
                        break
        sums = torch.zeros((clusters, dim), dtype=torch.float32, device=work.device)
        sums.index_add_(0, labels, work)
        counts = torch.bincount(labels, minlength=clusters)
        centroids = sums / counts.clamp_min(1).unsqueeze(1)
        if spherical:
            centroids = F.normalize(centroids, dim=-1, eps=1e-12)
        if torch.equal(labels, previous):
            break
    return centroids, labels, counts


def _contiguous_labels(tokens: int, block: int, device: torch.device) -> torch.Tensor:
    return torch.div(torch.arange(tokens, device=device), block, rounding_mode="floor")


def _qlocal_labels(query: torch.Tensor, clusters_per_block: int, seed: int) -> torch.Tensor:
    labels = torch.empty(query.shape[0], dtype=torch.long, device=query.device)
    offset = 0
    for block_id, start in enumerate(range(0, query.shape[0], 64)):
        part = query[start : start + 64]
        _, local, _ = _kmeans(
            part,
            min(clusters_per_block, part.shape[0]),
            iterations=3,
            seed=seed + block_id * 97,
        )
        labels[start : start + part.shape[0]] = local + offset
        offset += int(local.max()) + 1
    return labels


def _group_means(vectors: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    groups = int(labels.max()) + 1
    sums = torch.zeros((groups, vectors.shape[1]), dtype=torch.float32, device=vectors.device)
    sums.index_add_(0, labels, vectors.float())
    counts = torch.bincount(labels, minlength=groups)
    return sums / counts.clamp_min(1).unsqueeze(1), counts


def _expand_cluster_order(
    order: torch.Tensor,
    labels: torch.Tensor,
    budget: int,
) -> list[torch.Tensor]:
    rows = []
    for row in order:
        selected = []
        for cluster in row.tolist():
            members = torch.nonzero(labels == cluster, as_tuple=False).flatten()
            take = min(members.numel(), budget - len(selected))
            selected.extend(members[:take].tolist())
            if len(selected) == budget:
                break
        rows.append(torch.tensor(sorted(selected), dtype=torch.long, device=labels.device))
    return rows


def _fixed_cluster_selection(
    query: torch.Tensor,
    query_labels: torch.Tensor,
    key: torch.Tensor,
    spec: MethodSpec,
    budget: int,
    *,
    seed: int,
    radius_beta: float = 0.0,
    query_metric_rank: int | None = None,
    size_split_capacity: float | None = None,
) -> list[torch.Tensor]:
    q_centroids, _ = _group_means(query, query_labels)
    key_work = key
    query_work = q_centroids
    if query_metric_rank is not None:
        sample = F.normalize(query.float(), dim=-1)
        centered = sample - sample.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / max(1, sample.shape[0] - 1)
        _, eigenvectors = torch.linalg.eigh(covariance)
        basis = eigenvectors[:, -min(query_metric_rank, key.shape[1]) :]
        key_work = key.float() @ basis
        query_work = q_centroids.float() @ basis
    centroids, labels, counts = _kmeans(
        key_work,
        spec.k_clusters,
        iterations=spec.iterations,
        seed=seed,
    )
    if size_split_capacity is not None:
        capacity = max(1, int(math.ceil(key.shape[0] / spec.k_clusters * size_split_capacity)))
        next_label = int(labels.max()) + 1
        for cluster in range(next_label):
            members = torch.nonzero(labels == cluster, as_tuple=False).flatten()
            if members.numel() <= capacity:
                continue
            parts = math.ceil(members.numel() / capacity)
            _, local, _ = _kmeans(
                key_work.index_select(0, members),
                parts,
                iterations=spec.iterations,
                seed=seed + 1009 + cluster,
            )
            labels[members] = torch.where(local == 0, cluster, local + next_label - 1)
            next_label += parts - 1
        centroids, counts = _group_means(key_work, labels)
    scores = query_work.float() @ centroids.float().T / math.sqrt(query_work.shape[1])
    scores = scores + counts.float().clamp_min(1).log().unsqueeze(0)
    if radius_beta:
        normalized_key = F.normalize(key_work.float(), dim=-1)
        normalized_centroids = F.normalize(centroids.float(), dim=-1)
        assigned = normalized_centroids.index_select(0, labels)
        residual = 1.0 - (normalized_key * assigned).sum(dim=1)
        radii = torch.zeros(centroids.shape[0], device=key.device)
        radii.scatter_reduce_(0, labels, residual, reduce="amax", include_self=True)
        scores = scores + radius_beta * radii.unsqueeze(0)
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    return _expand_cluster_order(order, labels, budget)


def _temporal_selection(
    query: torch.Tensor,
    query_labels: torch.Tensor,
    key: torch.Tensor,
    frame_ids: torch.Tensor,
    spec: MethodSpec,
    budget: int,
    seed: int,
) -> list[torch.Tensor]:
    unique_frames = torch.unique(frame_ids, sorted=True)
    bins = min(spec.temporal_bins or 16, unique_frames.numel())
    frame_rank = torch.searchsorted(unique_frames, frame_ids)
    frame_bin = torch.div(frame_rank * bins, max(1, unique_frames.numel()), rounding_mode="floor")
    labels = torch.empty_like(frame_bin)
    offset = 0
    per_bin = max(1, math.ceil(spec.k_clusters / bins))
    for time_bin in range(bins):
        members = torch.nonzero(frame_bin == time_bin, as_tuple=False).flatten()
        if not members.numel():
            continue
        _, local, _ = _kmeans(
            key.index_select(0, members),
            min(per_bin, members.numel()),
            iterations=spec.iterations,
            seed=seed + time_bin * 101,
        )
        labels[members] = local + offset
        offset += int(local.max()) + 1
    centroids, counts = _group_means(key, labels)
    q_centroids, _ = _group_means(query, query_labels)
    scores = q_centroids.float() @ centroids.float().T / math.sqrt(key.shape[1])
    scores += counts.float().clamp_min(1).log().unsqueeze(0)
    return _expand_cluster_order(torch.argsort(scores, dim=1, descending=True), labels, budget)


def _paper_selection(
    query: torch.Tensor,
    query_labels: torch.Tensor,
    key: torch.Tensor,
    spec: MethodSpec,
    budget: int,
    seed: int,
) -> tuple[torch.Tensor, list[torch.Tensor], dict]:
    q_centroids, q_counts = _group_means(query, query_labels)
    metadata: dict = {
        "parameter_origin": spec.parameter_origin,
        "configured_q_clusters": spec.q_clusters,
        "configured_k_clusters": spec.k_clusters,
        "configured_top_p": spec.top_p,
    }
    if spec.name == "adacluster_ar":
        query_clusters = spec.q_clusters
        while True:
            q_centroids, query_labels, q_counts = _kmeans(
                query,
                query_clusters,
                iterations=spec.iterations,
                seed=seed,
                spherical=False,
            )
            q_residual = (
                query.float() - q_centroids.index_select(0, query_labels)
            ).norm(dim=1)
            if (
                float(torch.quantile(q_residual, 0.95))
                <= float(spec.query_threshold or 9.0)
                or query_clusters >= 600
            ):
                break
            query_clusters = min(600, query_clusters + 64)
        key_clusters = spec.k_clusters
        while True:
            k_centroids, k_labels, k_counts = _kmeans(
                key,
                key_clusters,
                iterations=spec.iterations,
                seed=seed + 1,
                spherical=False,
            )
            residual = (key.float() - k_centroids.index_select(0, k_labels)).norm(dim=1)
            if float(torch.quantile(residual, 0.95)) <= float(spec.threshold or 5.5) or key_clusters >= 600:
                break
            key_clusters = min(600, key_clusters + 64)
        positive, negative = q_centroids.clamp_min(0), q_centroids.clamp_max(0)
        minimum = torch.full_like(k_centroids, float("inf"))
        maximum = torch.full_like(k_centroids, -float("inf"))
        for cluster in range(k_centroids.shape[0]):
            members = key[k_labels == cluster].float()
            minimum[cluster] = members.min(dim=0).values
            maximum[cluster] = members.max(dim=0).values
        scores = positive @ maximum.T + negative @ minimum.T
        metadata["adaptive_k"] = key_clusters
        metadata["adaptive_q"] = query_clusters
    elif spec.name == "svoo_ar":
        q_centroids0, q_labels, _ = _kmeans(query, spec.q_clusters, iterations=spec.iterations, seed=seed)
        k_centroids0, k_labels, _ = _kmeans(key, spec.k_clusters, iterations=spec.iterations, seed=seed + 1)
        for iteration in range(spec.co_cluster_iterations or 2):
            q_centroids, _ = _group_means(query, q_labels)
            signatures = F.normalize(key.float() @ q_centroids.T, dim=-1)
            _, k_labels, _ = _kmeans(signatures, spec.k_clusters, iterations=spec.iterations, seed=seed + 101 * (iteration + 1))
            k_centroids, _ = _group_means(key, k_labels)
            signatures = F.normalize(query.float() @ k_centroids.T, dim=-1)
            _, q_labels, _ = _kmeans(signatures, spec.q_clusters, iterations=spec.iterations, seed=seed + 211 * (iteration + 1))
        query_labels = q_labels
        q_centroids, q_counts = _group_means(query, query_labels)
        k_centroids, k_counts = _group_means(key, k_labels)
        scores = q_centroids @ k_centroids.T / math.sqrt(key.shape[1])
        scores += k_counts.float().clamp_min(1).log().unsqueeze(0)
    elif spec.name == "scope_ar":
        q_centroids, query_labels, q_counts = _kmeans(
            query,
            spec.q_clusters,
            iterations=spec.iterations,
            seed=seed,
        )
        proxy = torch.zeros((q_centroids.shape[0], key.shape[0]), device=key.device)
        feature_slices = (slice(0, 44), slice(44, 86), slice(86, key.shape[1]))
        for subspace, feature_slice in enumerate(feature_slices):
            centroids, labels, _ = _kmeans(
                key[:, feature_slice],
                spec.k_clusters,
                iterations=spec.iterations,
                seed=seed + subspace,
            )
            proxy += (q_centroids[:, feature_slice].float() @ centroids.T).index_select(1, labels)
        scores = proxy / math.sqrt(key.shape[1])
        order = torch.argsort(scores, dim=1, descending=True, stable=True)
        metadata["q_clusters"] = q_centroids.shape[0]
        metadata["subspace_k_clusters"] = [
            min(spec.k_clusters, key[:, feature_slice].shape[0])
            for feature_slice in feature_slices
        ]
        return query_labels, [row[:budget].sort().values for row in order], metadata
    else:
        q_centroids, query_labels, q_counts = _kmeans(
            query,
            spec.q_clusters,
            iterations=spec.iterations,
            seed=seed,
        )
        k_centroids, k_labels, k_counts = _kmeans(key, spec.k_clusters, iterations=spec.iterations, seed=seed + 1)
        scores = q_centroids @ k_centroids.T / math.sqrt(key.shape[1])
        scores += k_counts.float().clamp_min(1).log().unsqueeze(0)
    if spec.name not in {"svoo_ar"}:
        k_labels = locals().get("k_labels")
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    metadata["q_clusters"] = q_centroids.shape[0]
    metadata["k_clusters"] = k_centroids.shape[0]
    return query_labels, _expand_cluster_order(order, k_labels, budget), metadata


def build_route_plan(
    *,
    method: str,
    routing_stage: str,
    query_labels: torch.Tensor,
    selections: list[list[list[torch.Tensor]]],
    history_frame_ids: torch.Tensor,
    history_token_ids: torch.Tensor,
    candidate_history_tokens: int,
    exact_k_tokens: int,
    density: float,
    metadata: dict,
) -> HistoryRoutePlan:
    query_labels = query_labels.clone()
    groups_before = 0
    groups_after = 0
    compacted: list[list[list[torch.Tensor]]] = [
        [[] for _ in range(query_labels.shape[1])]
        for _ in range(query_labels.shape[0])
    ]
    for batch_index in range(query_labels.shape[0]):
        for head in range(query_labels.shape[1]):
            rows = selections[batch_index][head]
            groups_before += len(rows)
            signatures: dict[bytes, int] = {}
            remap = torch.empty(len(rows), dtype=torch.long, device=query_labels.device)
            unique_rows = []
            for old_group, row in enumerate(rows):
                cpu = row.detach().to("cpu").contiguous()
                signature = cpu.numpy().tobytes()
                new_group = signatures.get(signature)
                if new_group is None:
                    new_group = len(unique_rows)
                    signatures[signature] = new_group
                    unique_rows.append(row)
                remap[old_group] = new_group
            query_labels[batch_index, head] = remap.index_select(
                0, query_labels[batch_index, head]
            )
            compacted[batch_index][head] = unique_rows
            groups_after += len(unique_rows)
    selections = compacted
    metadata = dict(metadata)
    metadata["query_groups_before_compaction"] = groups_before
    metadata["query_groups_after_compaction"] = groups_after
    batch, heads, query_tokens = query_labels.shape
    group_counts = torch.zeros((batch, heads, 1), dtype=torch.long)
    max_groups = max(int(query_labels.max()) + 1, 1)
    group_counts = torch.zeros((batch, heads, max_groups), dtype=torch.long)
    group_sizes = torch.zeros_like(group_counts)
    unions: list[list[torch.Tensor]] = [[None for _ in range(heads)] for _ in range(batch)]  # type: ignore
    group_maps: list[list[list[torch.Tensor]]] = [[[] for _ in range(heads)] for _ in range(batch)]
    max_union = 0
    max_selected = 0
    for b in range(batch):
        for h in range(heads):
            groups = int(query_labels[b, h].max()) + 1
            selected_rows = selections[b][h]
            union = torch.unique(torch.cat(selected_rows), sorted=True)
            unions[b][h] = union
            max_union = max(max_union, union.numel())
            lookup = torch.full((candidate_history_tokens,), -1, dtype=torch.long, device=union.device)
            lookup[union] = torch.arange(union.numel(), device=union.device)
            for group in range(groups):
                row = selected_rows[group]
                group_maps[b][h].append(lookup.index_select(0, row))
                group_counts[b, h, group] = row.numel()
                group_sizes[b, h, group] = int((query_labels[b, h] == group).sum())
                max_selected = max(max_selected, row.numel())
    union_frames = torch.full((batch, heads, max_union), -1, dtype=torch.long)
    union_tokens = torch.full_like(union_frames, -1)
    group_union = torch.full((batch, heads, max_groups, max_selected), -1, dtype=torch.long)
    for b in range(batch):
        for h in range(heads):
            union = unions[b][h]
            union_frames[b, h, : union.numel()] = history_frame_ids[b, h].to(union.device).index_select(0, union).cpu()
            union_tokens[b, h, : union.numel()] = history_token_ids[b, h].to(union.device).index_select(0, union).cpu()
            for group, row in enumerate(group_maps[b][h]):
                group_union[b, h, group, : row.numel()] = row.cpu()
    return HistoryRoutePlan(
        method=method,
        routing_stage=routing_stage,
        query_labels=query_labels.detach().cpu(),
        query_group_sizes=group_sizes,
        union_frame_ids=union_frames,
        union_token_ids=union_tokens,
        group_union_indices=group_union,
        group_history_counts=group_counts,
        candidate_history_tokens=candidate_history_tokens,
        query_tokens=query_tokens,
        exact_k_tokens=exact_k_tokens,
        target_history_density=density,
        metadata=metadata,
    )


def route_history(
    query: torch.Tensor,
    history_key: torch.Tensor,
    history_frame_ids: torch.Tensor,
    history_token_ids: torch.Tensor,
    *,
    method: str,
    density: float,
    exact_k_tokens: int,
    seed: int = 42,
    spec_override: dict | None = None,
) -> HistoryRoutePlan:
    """Route rectangular ``[B,Q,H,D]`` queries to ``[B,K,H,D]`` history."""

    spec = method_spec(method)
    if spec_override:
        allowed = set(spec.__dataclass_fields__)
        unknown = set(spec_override) - allowed
        if unknown:
            raise ValueError(f"unknown method override fields: {sorted(unknown)}")
        spec = replace(spec, **spec_override)
    if query.ndim != 4 or history_key.ndim != 4:
        raise ValueError("query/history_key must be [B,T,H,D]")
    batch, query_tokens, heads, _ = query.shape
    history_tokens = history_key.shape[1]
    if history_frame_ids.shape != (batch, heads, history_tokens):
        raise ValueError("history frame-id shape mismatch")
    budget = max(1, min(history_tokens, round(history_tokens * density)))
    all_labels = torch.empty((batch, heads, query_tokens), dtype=torch.long, device=query.device)
    all_selections: list[list[list[torch.Tensor]]] = [[[] for _ in range(heads)] for _ in range(batch)]
    metadata: dict = {"parameter_origin": spec.parameter_origin}
    for b in range(batch):
        for h in range(heads):
            q = query[b, :, h]
            k = history_key[b, :, h]
            local_seed = seed + b * 100003 + h * 1009
            if method in {"rag_dense", "random_history", "rag_local"}:
                labels = torch.zeros(query_tokens, dtype=torch.long, device=query.device)
                if method == "rag_dense":
                    rows = [torch.arange(history_tokens, device=query.device)]
                elif method == "rag_local":
                    rows = [torch.empty(0, dtype=torch.long, device=query.device)]
                else:
                    generator = torch.Generator(device="cpu").manual_seed(local_seed)
                    rows = [torch.randperm(history_tokens, generator=generator)[:budget].sort().values.to(query.device)]
            elif method in {"block64_history", "native_block"}:
                labels = _contiguous_labels(query_tokens, 64, query.device)
                q_centroids, _ = _group_means(q, labels)
                k_labels = _contiguous_labels(history_tokens, 64, k.device)
                k_centroids, _ = _group_means(k, k_labels)
                order = torch.argsort(q_centroids @ k_centroids.T, dim=1, descending=True)
                rows = _expand_cluster_order(order, k_labels, budget)
            elif method == "token_oracle":
                labels = _contiguous_labels(query_tokens, 64, query.device)
                q_centroids, _ = _group_means(q, labels)
                order = torch.argsort(q_centroids.float() @ k.float().T, dim=1, descending=True)
                rows = [row[:budget].sort().values for row in order]
            elif method == "qlocal_kmeans8_ar":
                labels = _qlocal_labels(q, 8, local_seed)
                block_spec = replace(spec, k_clusters=math.ceil(history_tokens / 64))
                rows = _fixed_cluster_selection(q, labels, k, block_spec, budget, seed=local_seed)
            elif method in {"kcluster32_history", "fixed_k128_history", "fixed_k256_history"}:
                labels = torch.zeros(query_tokens, dtype=torch.long, device=query.device)
                rows = _fixed_cluster_selection(q, labels, k, spec, budget, seed=local_seed)
            elif method == "radius_k256_ar":
                labels = _contiguous_labels(query_tokens, 64, query.device)
                rows = _fixed_cluster_selection(q, labels, k, spec, budget, seed=local_seed, radius_beta=float(spec.threshold or 0.5))
            elif method == "qmetric_k256_r32_ar":
                labels = _contiguous_labels(query_tokens, 64, query.device)
                rows = _fixed_cluster_selection(q, labels, k, spec, budget, seed=local_seed, query_metric_rank=spec.rank)
            elif method == "temporal_k256_t16_ar":
                labels = _contiguous_labels(query_tokens, 64, query.device)
                rows = _temporal_selection(q, labels, k, history_frame_ids[b, h].to(k.device), spec, budget, local_seed)
            elif method == "sizesplit_k128_c2_ar":
                labels = _contiguous_labels(query_tokens, 64, query.device)
                rows = _fixed_cluster_selection(q, labels, k, spec, budget, seed=local_seed, size_split_capacity=spec.capacity_factor)
            elif method in {"svg2_ar", "adacluster_ar", "svoo_ar", "scope_ar"}:
                labels, rows, method_metadata = _paper_selection(
                    q,
                    _contiguous_labels(query_tokens, 64, query.device),
                    k,
                    spec,
                    budget,
                    local_seed,
                )
                metadata.update(method_metadata)
            else:
                raise ValueError(f"method {method} has no history router")
            all_labels[b, h] = labels
            all_selections[b][h] = rows
    return build_route_plan(
        method=method,
        routing_stage=spec.routing_stage,
        query_labels=all_labels,
        selections=all_selections,
        history_frame_ids=history_frame_ids,
        history_token_ids=history_token_ids,
        candidate_history_tokens=history_tokens,
        exact_k_tokens=exact_k_tokens,
        density=density,
        metadata=metadata,
    )
