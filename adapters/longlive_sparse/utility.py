"""Causal Block64 value proxies, query membership, and marginal selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

from .ar_routing import build_route_plan
from .contexts import OnlineRoutingContext
from .route_plan import HistoryRoutePlan


VALUE_CANDIDATES = {
    "sum_value",
    "peak_value",
    "entropy_effective_groups",
    "count_uniform",
    "count_peak50",
}


def standardize_history_blocks(value: torch.Tensor) -> torch.Tensor:
    """Standardize along history blocks for fixed B/H/query-group axes."""

    mean = value.mean(dim=-1, keepdim=True)
    scale = value.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (value - mean) / scale


@dataclass(frozen=True)
class OnlineUtilityProxy:
    direct_scores: torch.Tensor
    direct_probabilities: torch.Tensor
    value_contribution_proxy: torch.Tensor
    combined_scores: torch.Tensor
    block_probabilities: torch.Tensor


def compute_online_utility_proxy(
    context: OnlineRoutingContext,
    *,
    log_prior: torch.Tensor | None = None,
) -> OnlineUtilityProxy:
    """Compute the frozen Final proxy using online-legal compact tensors only."""

    query = context.query_centroids.float()
    key = context.key_prototypes.float()
    value = context.value_prototypes.float()
    dim = query.shape[-1]
    direct = torch.einsum("bhgd,bhkd->bhgk", query, key) / math.sqrt(dim)
    if log_prior is not None:
        try:
            direct = direct + torch.broadcast_to(log_prior.float(), direct.shape)
        except RuntimeError as error:
            raise ValueError("log_prior is not broadcastable to B/H/G/K") from error
    probability = torch.softmax(direct, dim=-1)
    value_norm = torch.linalg.vector_norm(value, dim=-1).unsqueeze(2)
    contribution = probability * value_norm
    combined = standardize_history_blocks(direct) + standardize_history_blocks(
        contribution
    )
    block_probability = torch.softmax(combined, dim=-1)
    return OnlineUtilityProxy(
        direct_scores=direct,
        direct_probabilities=probability,
        value_contribution_proxy=contribution,
        combined_scores=combined,
        block_probabilities=block_probability,
    )


def aggregate_value_candidate(
    block_probabilities: torch.Tensor, candidate: str
) -> torch.Tensor:
    """Aggregate B/H/G/K probabilities into one causal value per B/H/K."""

    if candidate not in VALUE_CANDIDATES:
        raise ValueError(f"unsupported value candidate: {candidate!r}")
    if block_probabilities.ndim != 4:
        raise ValueError("block_probabilities must be [B,H,G,K]")
    peak = block_probabilities.amax(dim=2)
    if candidate == "sum_value":
        return block_probabilities.sum(dim=2)
    if candidate == "peak_value":
        return peak
    if candidate == "entropy_effective_groups":
        across_groups = block_probabilities / block_probabilities.sum(
            dim=2, keepdim=True
        ).clamp_min(1e-12)
        entropy = -(
            across_groups.clamp_min(1e-12)
            * across_groups.clamp_min(1e-12).log()
        ).sum(dim=2)
        return peak * entropy.exp()
    if candidate == "count_uniform":
        blocks = block_probabilities.shape[-1]
        count = (block_probabilities >= 1.0 / blocks).sum(dim=2)
        return peak * count.to(peak.dtype)
    group_peak = block_probabilities.amax(dim=-1, keepdim=True)
    count = (block_probabilities >= 0.5 * group_peak).sum(dim=2)
    return peak * count.to(peak.dtype)


def select_group_membership(
    scores: torch.Tensor,
    union_mask: torch.Tensor,
    *,
    policy: str,
    top_p: float = 0.90,
    min_k_ratio: float = 0.10,
) -> torch.Tensor:
    """Choose per-query membership without changing the bounded union itself."""

    if scores.ndim != 4:
        raise ValueError("scores must be [B,H,G,K]")
    if union_mask.shape != scores.shape[:2] + scores.shape[3:]:
        raise ValueError("union_mask must be [B,H,K]")
    if union_mask.dtype != torch.bool:
        raise ValueError("union_mask must be boolean")
    expanded = union_mask.unsqueeze(2).expand_as(scores)
    if policy == "legacy_exact_union":
        return expanded.clone()
    if policy != "mass_preserving_top_p":
        raise ValueError(f"unsupported group membership policy: {policy!r}")
    if not 0.0 < top_p <= 1.0 or not 0.0 <= min_k_ratio <= 1.0:
        raise ValueError("invalid Top-p or minimum-K ratio")

    masked_scores = scores.masked_fill(~expanded, -float("inf"))
    probabilities = torch.softmax(masked_scores, dim=-1)
    probabilities = torch.where(expanded, probabilities, torch.zeros_like(probabilities))
    order = torch.argsort(probabilities, dim=-1, descending=True, stable=True)
    ordered_probability = probabilities.gather(-1, order)
    cumulative = ordered_probability.cumsum(dim=-1)
    selected_order = cumulative - ordered_probability < top_p
    union_counts = union_mask.sum(dim=-1, keepdim=True)
    minimum = torch.ceil(union_counts.float() * min_k_ratio).long().clamp_min(1)
    ranks = torch.arange(scores.shape[-1], device=scores.device).view(1, 1, 1, -1)
    selected_order |= ranks < minimum.unsqueeze(2)
    selected = torch.zeros_like(expanded)
    selected.scatter_(-1, order, selected_order)
    return selected & expanded


def query_reuse_statistics(membership: torch.Tensor) -> dict[str, float]:
    if membership.ndim != 4 or membership.dtype != torch.bool:
        raise ValueError("membership must be boolean [B,H,G,K]")
    uses = membership.sum(dim=2).float()
    active = uses > 0
    if not bool(active.any()):
        return {
            "reuse_unit": "route_union_item",
            "active_union_items": 0.0,
            "query_groups_per_block_mean": 0.0,
            "query_groups_per_block_p95": 0.0,
            "query_groups_per_block_max": 0.0,
        }
    values = uses[active]
    return {
        "reuse_unit": "route_union_item",
        "active_union_items": float(active.sum()),
        "query_groups_per_block_mean": float(values.mean()),
        "query_groups_per_block_p95": float(torch.quantile(values, 0.95)),
        "query_groups_per_block_max": float(values.max()),
    }


def route_plan_membership(plan: HistoryRoutePlan) -> torch.Tensor:
    """Expand compact group-union indices into a logical B/H/G/U mask."""

    batch, heads, groups = plan.group_history_counts.shape
    union_width = plan.union_frame_ids.shape[-1]
    membership = torch.zeros((batch, heads, groups, union_width), dtype=torch.bool)
    for batch_index in range(batch):
        for head_index in range(heads):
            valid_union = int((plan.union_frame_ids[batch_index, head_index] >= 0).sum())
            for group_index in range(groups):
                count = int(plan.group_history_counts[batch_index, head_index, group_index])
                indices = plan.group_union_indices[
                    batch_index, head_index, group_index, :count
                ].detach().to("cpu")
                if count and int(indices.max()) >= valid_union:
                    raise IndexError("route plan membership references padded union")
                membership[batch_index, head_index, group_index, indices] = True
    return membership


def apply_query_group_policy(
    plan: HistoryRoutePlan,
    context: OnlineRoutingContext,
    *,
    policy: str,
    top_p: float = 0.90,
    min_k_ratio: float = 0.10,
) -> HistoryRoutePlan:
    """Change per-group membership while preserving the exact physical union."""

    candidate_frames = tuple(
        int(value)
        for value in context.metadata.get(
            "candidate_frame_ids",
            tuple(dict.fromkeys(int(value) for value in context.block_frame_ids)),
        )
    )
    if not candidate_frames or len(candidate_frames) != len(set(candidate_frames)):
        raise ValueError("candidate frame ids must be non-empty and unique")
    frame_tokens = int(context.block_token_ends.max())
    frame_to_rank = {frame: rank for rank, frame in enumerate(candidate_frames)}
    batch, heads, union_width = plan.union_frame_ids.shape
    if context.query_centroids.shape[:2] != (batch, heads):
        raise ValueError("route and online context must share batch/head axes")
    group_sizes = context.query_group_sizes.long()
    query_totals = group_sizes.sum(dim=-1)
    if bool((query_totals != plan.query_tokens).any()):
        raise ValueError("online query groups must cover the route query length")
    target_query_labels = torch.empty(
        (batch, heads, plan.query_tokens), dtype=torch.long
    )
    for batch_index in range(batch):
        for head_index in range(heads):
            target_query_labels[batch_index, head_index] = torch.repeat_interleave(
                torch.arange(group_sizes.shape[-1], dtype=torch.long),
                group_sizes[batch_index, head_index],
            )
    proxy = compute_online_utility_proxy(context)
    union_block_ids = torch.full((batch, heads, union_width), -1, dtype=torch.long)
    union_dense_indices = torch.full_like(union_block_ids, -1)
    selected_blocks = torch.zeros((batch, heads, context.blocks), dtype=torch.bool)
    original_coordinates: list[list[set[tuple[int, int]]]] = [
        [set() for _ in range(heads)] for _ in range(batch)
    ]
    for batch_index in range(batch):
        for head_index in range(heads):
            for union_index in range(union_width):
                frame = int(plan.union_frame_ids[batch_index, head_index, union_index])
                token = int(plan.union_token_ids[batch_index, head_index, union_index])
                if frame < 0:
                    continue
                matches = torch.nonzero(
                    (context.block_frame_ids == frame)
                    & (context.block_token_starts <= token)
                    & (context.block_token_ends > token),
                    as_tuple=False,
                ).flatten()
                if matches.numel() != 1:
                    raise KeyError("each route coordinate must map to one Block64 prototype")
                block = int(matches[0])
                union_block_ids[batch_index, head_index, union_index] = block
                selected_blocks[batch_index, head_index, block] = True
                union_dense_indices[batch_index, head_index, union_index] = (
                    frame_to_rank[frame] * frame_tokens + token
                )
                original_coordinates[batch_index][head_index].add((frame, token))
    membership = select_group_membership(
        proxy.combined_scores,
        selected_blocks,
        policy=policy,
        top_p=top_p,
        min_k_ratio=min_k_ratio,
    )
    for batch_index in range(batch):
        for head_index in range(heads):
            active_mask = group_sizes[batch_index, head_index] > 0
            membership[batch_index, head_index, ~active_mask] = False
            for block in torch.nonzero(
                selected_blocks[batch_index, head_index], as_tuple=False
            ).flatten():
                if not bool(
                    membership[
                        batch_index, head_index, active_mask, block
                    ].any()
                ):
                    active_groups = torch.nonzero(
                        active_mask, as_tuple=False
                    ).flatten()
                    best_group = int(
                        active_groups[
                            proxy.combined_scores[
                                batch_index, head_index, active_mask, block
                            ].argmax()
                        ]
                    )
                    membership[batch_index, head_index, best_group, block] = True
    selections: list[list[list[torch.Tensor]]] = [
        [[] for _ in range(heads)] for _ in range(batch)
    ]
    for batch_index in range(batch):
        for head_index in range(heads):
            groups = group_sizes.shape[-1]
            for group in range(groups):
                valid = union_block_ids[batch_index, head_index] >= 0
                include = valid & membership[
                    batch_index,
                    head_index,
                ].index_select(
                    -1, union_block_ids[batch_index, head_index].clamp_min(0)
                )[group]
                selections[batch_index][head_index].append(
                    union_dense_indices[batch_index, head_index, include]
                    .sort()
                    .values
                )
    candidate_tokens = len(candidate_frames) * frame_tokens
    history_frames = torch.tensor(candidate_frames).repeat_interleave(frame_tokens)
    history_tokens = torch.arange(frame_tokens).repeat(len(candidate_frames))
    history_frames = history_frames.view(1, 1, -1).expand(batch, heads, -1)
    history_tokens = history_tokens.view(1, 1, -1).expand(batch, heads, -1)
    result = build_route_plan(
        method="query_policy_history",
        routing_stage=plan.routing_stage,
        query_labels=target_query_labels,
        selections=selections,
        history_frame_ids=history_frames,
        history_token_ids=history_tokens,
        candidate_history_tokens=candidate_tokens,
        exact_k_tokens=plan.exact_k_tokens,
        density=plan.target_history_density,
        metadata={
            **plan.metadata,
            "source_route_plan_sha256": plan.digest(),
            "group_selection_policy": policy,
            "group_top_p": top_p,
            "group_min_k_ratio": min_k_ratio,
            "physical_union_preserved": True,
        },
    )
    for batch_index in range(batch):
        for head_index in range(heads):
            result_coordinates = {
                (int(frame), int(token))
                for frame, token in zip(
                    result.union_frame_ids[batch_index, head_index],
                    result.union_token_ids[batch_index, head_index],
                )
                if int(frame) >= 0
            }
            if result_coordinates != original_coordinates[batch_index][head_index]:
                raise RuntimeError("query membership policy changed the physical union")
    return result


def greedy_marginal_select(
    benefits: torch.Tensor,
    *,
    budget: int,
    set_cost: Callable[[torch.Tensor], float],
) -> torch.Tensor:
    """Deterministic set-marginal greedy selection for one block vector."""

    if benefits.ndim != 1:
        raise ValueError("benefits must be one-dimensional")
    if not 0 <= budget <= benefits.numel():
        raise ValueError("budget must be between zero and the block count")
    selected = torch.zeros(benefits.numel(), dtype=torch.bool)
    current_cost = float(set_cost(selected.clone()))
    for _ in range(budget):
        best_index = None
        best_ratio = -float("inf")
        for index in range(benefits.numel()):
            if bool(selected[index]):
                continue
            candidate = selected.clone()
            candidate[index] = True
            marginal_cost = max(float(set_cost(candidate)) - current_cost, 1e-12)
            ratio = float(benefits[index]) / marginal_cost
            if ratio > best_ratio or (
                ratio == best_ratio and (best_index is None or index < best_index)
            ):
                best_ratio = ratio
                best_index = index
        if best_index is None:
            raise RuntimeError("marginal selector could not fill its budget")
        selected[best_index] = True
        current_cost = float(set_cost(selected.clone()))
    return selected
