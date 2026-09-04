"""Causal Block64 value proxies, query membership, and marginal selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

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
            "active_union_blocks": 0.0,
            "query_groups_per_block_mean": 0.0,
            "query_groups_per_block_p95": 0.0,
            "query_groups_per_block_max": 0.0,
        }
    values = uses[active]
    return {
        "active_union_blocks": float(active.sum()),
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
