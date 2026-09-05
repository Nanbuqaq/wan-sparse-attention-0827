"""Causal system-utility admission that produces a new HistoryRoutePlan."""

from __future__ import annotations

import math
import hashlib
from dataclasses import asdict, dataclass
from typing import Callable

import torch

from .ar_routing import build_route_plan
from .contexts import OnlineRoutingContext
from .cost_model import CausalPipelineState, SystemCostModel
from .route_plan import HistoryRoutePlan
from .transfer_plan import build_transfer_plan
from .utility import (
    VALUE_CANDIDATES,
    aggregate_value_candidate,
    compute_online_utility_proxy,
    select_group_membership,
)


SetCost = Callable[[torch.Tensor], float]
SetCostFactory = Callable[[int, int], SetCost]


@dataclass(frozen=True)
class SystemUtilityRouteConfig:
    value_candidate: str
    cost_strategy: str
    history_density: float = 0.25
    correlation_fraction: float = 0.70
    coverage_fraction: float = 0.15
    remote_fraction: float = 0.15
    exploration_fraction: float = 0.0
    remote_min_age: int = 2
    group_selection_policy: str = "legacy_exact_union"
    group_top_p: float = 0.90
    group_min_k_ratio: float = 0.10

    def __post_init__(self) -> None:
        if self.value_candidate not in VALUE_CANDIDATES:
            raise ValueError(f"unsupported value candidate: {self.value_candidate!r}")
        if self.cost_strategy not in {"static_block", "marginal_set"}:
            raise ValueError(f"unsupported cost strategy: {self.cost_strategy!r}")
        if not 0.0 < self.history_density <= 1.0:
            raise ValueError("history_density must be in (0,1]")
        fractions = (
            self.correlation_fraction,
            self.coverage_fraction,
            self.remote_fraction,
            self.exploration_fraction,
        )
        if any(value < 0 for value in fractions) or sum(fractions) > 1.0 + 1e-9:
            raise ValueError("route budget fractions must be non-negative and sum to at most one")
        if self.remote_min_age < 0:
            raise ValueError("remote_min_age must be non-negative")
        if self.group_selection_policy not in {
            "legacy_exact_union",
            "mass_preserving_top_p",
        }:
            raise ValueError("unsupported group selection policy")
        if not 0.0 < self.group_top_p <= 1.0:
            raise ValueError("group_top_p must be in (0,1]")
        if not 0.0 <= self.group_min_k_ratio <= 1.0:
            raise ValueError("group_min_k_ratio must be in [0,1]")

    def as_dict(self) -> dict:
        return asdict(self)


def _candidate_frames(context: OnlineRoutingContext) -> tuple[int, ...]:
    declared = context.metadata.get("candidate_frame_ids")
    if declared:
        result = tuple(int(value) for value in declared)
    else:
        result = tuple(dict.fromkeys(int(value) for value in context.block_frame_ids))
    if not result or len(result) != len(set(result)):
        raise ValueError("candidate frame ids must be non-empty and unique")
    if set(result) != set(int(value) for value in context.block_frame_ids):
        raise ValueError("candidate frame ids do not match block metadata")
    return result


def _query_labels(group_sizes: torch.Tensor) -> torch.Tensor:
    batch, heads, groups = group_sizes.shape
    totals = group_sizes.sum(dim=-1)
    if bool((totals != totals.reshape(-1)[0]).any()):
        raise ValueError("all batch/head query group sizes must sum to one Q length")
    query_tokens = int(totals.reshape(-1)[0])
    labels = torch.empty((batch, heads, query_tokens), dtype=torch.long)
    for batch_index in range(batch):
        for head_index in range(heads):
            labels[batch_index, head_index] = torch.repeat_interleave(
                torch.arange(groups, dtype=torch.long),
                group_sizes[batch_index, head_index].long(),
            )
    return labels


def _select_scored(
    selected: torch.Tensor,
    scores: torch.Tensor,
    widths: torch.Tensor,
    *,
    token_budget: int,
    allowed: torch.Tensor | None,
    set_cost: SetCost | None,
) -> tuple[torch.Tensor, int]:
    selected = selected.clone()
    admitted = 0
    if set_cost is None:
        # Static byte cost is independent of S. Sorting once gives exactly the
        # old repeated-argmax order, including stable ties and short frame tails.
        ratios = scores.double() / widths.double()
        order = torch.argsort(ratios, descending=True, stable=True).tolist()
        for index in order:
            width = int(widths[index])
            if bool(selected[index]) or width > token_budget - admitted:
                continue
            if allowed is not None and not bool(allowed[index]):
                continue
            selected[index] = True
            admitted += width
        return selected, admitted
    while admitted < token_budget:
        best_index = None
        best_ratio = -float("inf")
        current_cost = float(set_cost(selected)) if set_cost is not None else 0.0
        for index in range(selected.numel()):
            width = int(widths[index])
            if bool(selected[index]) or width > token_budget - admitted:
                continue
            if allowed is not None and not bool(allowed[index]):
                continue
            if set_cost is None:
                marginal_cost = float(width)
            else:
                candidate = selected.clone()
                candidate[index] = True
                marginal_cost = max(float(set_cost(candidate)) - current_cost, 1e-12)
            ratio = float(scores[index]) / marginal_cost
            if ratio > best_ratio or (
                ratio == best_ratio and (best_index is None or index < best_index)
            ):
                best_ratio = ratio
                best_index = index
        if best_index is None:
            break
        selected[best_index] = True
        admitted += int(widths[best_index])
    return selected, admitted


def _coverage_scores(
    context: OnlineRoutingContext,
    selected: torch.Tensor,
) -> torch.Tensor:
    frames = context.block_frame_ids.float()
    centers = (
        context.block_token_starts.float() + context.block_token_ends.float() - 1
    ) / 2
    frame_span = (frames.max() - frames.min()).clamp_min(1.0)
    token_span = centers.max().clamp_min(1.0)
    points = torch.stack(
        ((frames - frames.min()) / frame_span, centers / token_span), dim=-1
    )
    if bool(selected.any()):
        distance = torch.cdist(points, points[selected]).amin(dim=-1)
    else:
        distance = torch.ones(points.shape[0])
    recency = 1.0 - context.block_age.float() / context.block_age.max().clamp_min(1.0)
    return distance + 0.05 * recency


def _select_coverage(
    context: OnlineRoutingContext,
    selected: torch.Tensor,
    widths: torch.Tensor,
    *,
    token_budget: int,
    set_cost: SetCost | None,
) -> tuple[torch.Tensor, int]:
    admitted = 0
    selected = selected.clone()
    while admitted < token_budget:
        scores = _coverage_scores(context, selected)
        updated, gained = _select_scored(
            selected,
            scores,
            widths,
            token_budget=min(
                token_budget - admitted,
                int(widths[~selected].max()) if bool((~selected).any()) else 0,
            ),
            allowed=None,
            set_cost=set_cost,
        )
        if gained == 0:
            break
        selected = updated
        admitted += gained
    return selected, admitted


def _history_coordinate_tables(
    context: OnlineRoutingContext,
) -> tuple[
    tuple[int, ...],
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor],
]:
    candidate_frames = _candidate_frames(context)
    frame_tokens = int(context.block_token_ends.max())
    if frame_tokens < 1:
        raise ValueError("frame token count must be positive")
    frame_to_rank = {frame: rank for rank, frame in enumerate(candidate_frames)}
    widths = context.block_token_ends.long() - context.block_token_starts.long()
    block_token_indices = []
    for block in range(context.blocks):
        frame = int(context.block_frame_ids[block])
        start = int(context.block_token_starts[block])
        end = int(context.block_token_ends[block])
        block_token_indices.append(
            torch.arange(
                frame_to_rank[frame] * frame_tokens + start,
                frame_to_rank[frame] * frame_tokens + end,
                dtype=torch.long,
            )
        )
    history_frames = torch.tensor(candidate_frames, dtype=torch.long).repeat_interleave(
        frame_tokens
    )
    history_tokens = torch.arange(frame_tokens, dtype=torch.long).repeat(
        len(candidate_frames)
    )
    return candidate_frames, frame_tokens, widths, history_frames, history_tokens, block_token_indices


def build_system_utility_route(
    context: OnlineRoutingContext,
    *,
    exact_k_tokens: int,
    config: SystemUtilityRouteConfig,
    set_cost_factory: SetCostFactory | None = None,
    log_prior: torch.Tensor | None = None,
) -> HistoryRoutePlan:
    """Build a causal HistoryRoutePlan under one physical shared-union cap."""

    if config.cost_strategy == "marginal_set" and set_cost_factory is None:
        raise ValueError("marginal_set routing requires a frozen set cost factory")
    proxy = compute_online_utility_proxy(context, log_prior=log_prior)
    main_scores = aggregate_value_candidate(
        proxy.block_probabilities, config.value_candidate
    )
    remote_scores = proxy.value_contribution_proxy.sum(dim=2)
    if context.past_attention_score is None:
        exploration_scores = context.block_age.float().view(1, 1, -1).expand_as(
            main_scores
        )
    else:
        past = torch.broadcast_to(
            context.past_attention_score.float(), main_scores.shape
        )
        exploration_scores = -past + context.block_age.float().view(1, 1, -1)
    (
        candidate_frames,
        frame_tokens,
        widths,
        history_frames_1d,
        history_tokens_1d,
        block_token_indices,
    ) = _history_coordinate_tables(context)
    candidate_history_tokens = len(candidate_frames) * frame_tokens
    target_tokens = max(
        1, math.floor(candidate_history_tokens * config.history_density)
    )
    fractions = (
        config.correlation_fraction,
        config.coverage_fraction,
        config.remote_fraction,
        config.exploration_fraction,
    )
    tier_budgets = [math.floor(target_tokens * value) for value in fractions]
    batch, heads, _ = main_scores.shape
    selected_blocks = torch.zeros((batch, heads, context.blocks), dtype=torch.bool)
    tier_admitted = torch.zeros((batch, heads, 4), dtype=torch.long)
    remote_allowed = context.block_age >= config.remote_min_age
    for batch_index in range(batch):
        for head_index in range(heads):
            selected = selected_blocks[batch_index, head_index]
            set_cost = (
                set_cost_factory(batch_index, head_index)
                if set_cost_factory is not None
                else None
            )
            selected, gained = _select_scored(
                selected,
                main_scores[batch_index, head_index],
                widths,
                token_budget=tier_budgets[0],
                allowed=None,
                set_cost=set_cost,
            )
            tier_admitted[batch_index, head_index, 0] = gained
            selected, gained = _select_coverage(
                context,
                selected,
                widths,
                token_budget=tier_budgets[1],
                set_cost=set_cost,
            )
            tier_admitted[batch_index, head_index, 1] = gained
            selected, gained = _select_scored(
                selected,
                remote_scores[batch_index, head_index],
                widths,
                token_budget=tier_budgets[2],
                allowed=remote_allowed,
                set_cost=set_cost,
            )
            tier_admitted[batch_index, head_index, 2] = gained
            selected, gained = _select_scored(
                selected,
                exploration_scores[batch_index, head_index],
                widths,
                token_budget=tier_budgets[3],
                allowed=None,
                set_cost=set_cost,
            )
            tier_admitted[batch_index, head_index, 3] = gained
            used = int(widths[selected].sum())
            selected, _ = _select_scored(
                selected,
                main_scores[batch_index, head_index],
                widths,
                token_budget=max(0, target_tokens - used),
                allowed=None,
                set_cost=set_cost,
            )
            if int(widths[selected].sum()) > target_tokens:
                raise RuntimeError("system utility route exceeded its physical union cap")
            selected_blocks[batch_index, head_index] = selected

    membership = select_group_membership(
        proxy.combined_scores,
        selected_blocks,
        policy=config.group_selection_policy,
        top_p=config.group_top_p,
        min_k_ratio=config.group_min_k_ratio,
    )
    for batch_index in range(batch):
        for head_index in range(heads):
            for block in torch.nonzero(
                selected_blocks[batch_index, head_index], as_tuple=False
            ).flatten():
                if not bool(membership[batch_index, head_index, :, block].any()):
                    group = int(
                        proxy.combined_scores[
                            batch_index, head_index, :, block
                        ].argmax()
                    )
                    membership[batch_index, head_index, group, block] = True

    query_labels = _query_labels(context.query_group_sizes)
    groups = context.query_group_sizes.shape[-1]
    selections: list[list[list[torch.Tensor]]] = [
        [[] for _ in range(heads)] for _ in range(batch)
    ]
    for batch_index in range(batch):
        for head_index in range(heads):
            for group in range(groups):
                blocks = torch.nonzero(
                    membership[batch_index, head_index, group], as_tuple=False
                ).flatten()
                tokens = [block_token_indices[int(block)] for block in blocks]
                selections[batch_index][head_index].append(
                    torch.cat(tokens).sort().values
                    if tokens
                    else torch.empty(0, dtype=torch.long)
                )
    history_frames = history_frames_1d.view(1, 1, -1).expand(batch, heads, -1)
    history_tokens = history_tokens_1d.view(1, 1, -1).expand(batch, heads, -1)
    metadata = {
        "online_context_only": True,
        "offline_teacher_used": False,
        "physical_union_cap_tokens": target_tokens,
        "selected_block_counts": selected_blocks.sum(dim=-1).tolist(),
        "selected_union_tokens": (
            selected_blocks * widths.view(1, 1, -1)
        ).sum(dim=-1).tolist(),
        "tier_budget_tokens": tier_budgets,
        "tier_admitted_tokens": tier_admitted.tolist(),
        "candidate_frame_ids": list(candidate_frames),
        "frame_tokens": frame_tokens,
        "route_config": config.as_dict(),
        "routing_identity": {"config": config.as_dict(),
            "role_prior_sha": hashlib.sha256(log_prior.detach().cpu().float().numpy().tobytes()).hexdigest()
                              if log_prior is not None else None},
        "hardware_profile_id": context.hardware_profile_id,
        "cost_model_version": context.cost_model_version,
        "causal_role_prior_used": log_prior is not None,
    }
    return build_route_plan(
        method="system_utility_history",
        routing_stage="pre-transfer",
        query_labels=query_labels,
        selections=selections,
        history_frame_ids=history_frames,
        history_token_ids=history_tokens,
        candidate_history_tokens=candidate_history_tokens,
        exact_k_tokens=exact_k_tokens,
        density=config.history_density,
        metadata=metadata,
    )


def build_cost_model_set_cost_factory(
    context: OnlineRoutingContext,
    model: SystemCostModel,
    *,
    exact_k_tokens: int,
    transfer_layout: str,
    transfer_mode: str,
    execution_dataflow: str,
    pipeline_state: CausalPipelineState | None = None,
) -> SetCostFactory:
    """Adapt a frozen SystemCostModel to deterministic set-marginal admission."""

    (
        candidate_frames,
        frame_tokens,
        _,
        history_frames,
        history_tokens,
        block_token_indices,
    ) = _history_coordinate_tables(context)
    bytes_per_token = int(
        context.metadata.get(
            "bytes_per_history_token",
            2
            * context.key_prototypes.shape[-1]
            * context.key_prototypes.element_size(),
        )
    )
    caches: dict[tuple[int, int], dict[bytes, float]] = {}

    def factory(batch_index: int, head_index: int) -> SetCost:
        cache = caches.setdefault((batch_index, head_index), {})

        def set_cost(mask: torch.Tensor) -> float:
            signature = mask.detach().to("cpu").contiguous().numpy().tobytes()
            if signature in cache:
                return cache[signature]
            blocks = torch.nonzero(mask, as_tuple=False).flatten()
            token_parts = [block_token_indices[int(block)] for block in blocks]
            selected = (
                torch.cat(token_parts).sort().values
                if token_parts
                else torch.empty(0, dtype=torch.long)
            )
            query_labels = torch.zeros((1, 1, 1), dtype=torch.long)
            route = build_route_plan(
                method="system_utility_cost_probe",
                routing_stage="pre-transfer",
                query_labels=query_labels,
                selections=[[[selected]]],
                history_frame_ids=history_frames.view(1, 1, -1),
                history_token_ids=history_tokens.view(1, 1, -1),
                candidate_history_tokens=len(candidate_frames) * frame_tokens,
                exact_k_tokens=exact_k_tokens,
                density=1.0,
                metadata={"cost_probe": True},
            )
            transfer = build_transfer_plan(
                route,
                candidate_frames,
                frame_tokens=frame_tokens,
                layout=transfer_layout,
                bytes_per_token=bytes_per_token,
            )
            prediction = model.predict(
                route,
                transfer,
                execution_dataflow=execution_dataflow,
                transfer_mode=transfer_mode,
                pipeline_state=pipeline_state,
            )
            cache[signature] = prediction.predicted_exposed_wait_s
            return cache[signature]

        return set_cost

    return factory
