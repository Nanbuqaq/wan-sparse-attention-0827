from __future__ import annotations

import pytest
import torch

from adapters.longlive_sparse.contexts import OnlineRoutingContext
from adapters.longlive_sparse.tethermem import soft_region_age_prior, solve_context_weight
from adapters.longlive_sparse.utility import (
    aggregate_value_candidate,
    compute_online_utility_proxy,
    greedy_marginal_select,
    query_reuse_statistics,
    route_plan_membership,
    select_group_membership,
    standardize_history_blocks,
)


def context() -> OnlineRoutingContext:
    generator = torch.Generator().manual_seed(4)
    return OnlineRoutingContext(
        query_centroids=torch.randn(1, 2, 3, 4, generator=generator),
        query_group_sizes=torch.ones(1, 2, 3, dtype=torch.long),
        key_prototypes=torch.randn(1, 2, 5, 4, generator=generator),
        value_prototypes=torch.randn(1, 2, 5, 4, generator=generator),
        block_frame_ids=torch.tensor([1, 1, 2, 2, 3]),
        block_token_starts=torch.tensor([0, 64, 0, 64, 0]),
        block_token_ends=torch.tensor([64, 128, 64, 128, 64]),
        block_age=torch.tensor([2, 2, 1, 1, 0]).float(),
    )


def test_standardization_is_per_query_group_along_blocks() -> None:
    value = torch.tensor([[[[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]]]])
    result = standardize_history_blocks(value)
    assert result[0, 0, 0].mean() == pytest.approx(0.0)
    assert torch.equal(result[0, 0, 1], torch.zeros(3))


def test_online_proxy_and_all_registered_aggregates_are_finite() -> None:
    proxy = compute_online_utility_proxy(context())
    assert torch.allclose(
        proxy.block_probabilities.sum(dim=-1),
        torch.ones_like(proxy.block_probabilities.sum(dim=-1)),
    )
    for candidate in (
        "sum_value",
        "peak_value",
        "entropy_effective_groups",
        "count_uniform",
        "count_peak50",
    ):
        value = aggregate_value_candidate(proxy.block_probabilities, candidate)
        assert value.shape == (1, 2, 5)
        assert torch.isfinite(value).all()


def test_mass_preserving_membership_stays_inside_union() -> None:
    scores = torch.tensor([[[[4.0, 3.0, 2.0, 1.0], [1.0, 4.0, 3.0, 2.0]]]])
    union = torch.tensor([[[True, True, True, False]]])
    exact = select_group_membership(
        scores, union, policy="legacy_exact_union"
    )
    top_p = select_group_membership(
        scores,
        union,
        policy="mass_preserving_top_p",
        top_p=0.8,
        min_k_ratio=0.1,
    )
    assert exact.sum() == 6
    assert top_p.sum() < exact.sum()
    assert not bool((top_p & ~union.unsqueeze(2)).any())
    stats = query_reuse_statistics(top_p)
    assert stats["active_union_items"] > 0


def test_route_plan_membership_matches_compact_indices() -> None:
    from adapters.longlive_sparse.route_plan import HistoryRoutePlan

    plan = HistoryRoutePlan(
        method="block64_history",
        routing_stage="pre-transfer",
        query_labels=torch.tensor([[[0, 0, 1, 1]]]),
        query_group_sizes=torch.tensor([[[2, 2]]]),
        union_frame_ids=torch.tensor([[[5, 5, 7, -1]]]),
        union_token_ids=torch.tensor([[[1, 2, 6, -1]]]),
        group_union_indices=torch.tensor([[[[0, 1], [1, 2]]]]),
        group_history_counts=torch.tensor([[[2, 2]]]),
        candidate_history_tokens=16,
        query_tokens=4,
        exact_k_tokens=8,
        target_history_density=0.25,
    )
    membership = route_plan_membership(plan)
    assert membership.shape == (1, 1, 2, 4)
    assert membership[0, 0, 0].tolist() == [True, True, False, False]
    assert membership[0, 0, 1].tolist() == [False, True, True, False]


def test_marginal_selector_rewards_adjacent_blocks() -> None:
    benefits = torch.tensor([3.0, 2.9, 2.8, 1.0])

    def cost(mask: torch.Tensor) -> float:
        indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        if not indices:
            return 0.0
        runs = 1 + sum(b != a + 1 for a, b in zip(indices, indices[1:]))
        return float(len(indices) + 2 * runs)

    selected = greedy_marginal_select(benefits, budget=2, set_cost=cost)
    assert selected.tolist() == [True, True, False, False]


def test_soft_tether_prior_matches_hard_region_table() -> None:
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    history = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    age = torch.tensor([[[0.25, 0.5]]])
    prior = soft_region_age_prior(query, history, age, context_weight=0.2)
    assert prior[0, 0, 0].tolist() == pytest.approx([1.0, 0.2])
    assert prior[0, 0, 1].tolist() == pytest.approx([0.2, 0.5])
    assert solve_context_weight(0.1) == pytest.approx(1.0 / 6.0)


def test_tether_prior_can_modify_online_proxy_without_teacher_inputs() -> None:
    current = context()
    query_roles = torch.tensor([[[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]])
    history_roles = torch.tensor(
        [[
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        ]]
    )
    age = torch.ones(1, 2, 5)
    prior = soft_region_age_prior(query_roles, history_roles, age, context_weight=0.2)
    proxy = compute_online_utility_proxy(current, log_prior=prior.log())
    assert torch.isfinite(proxy.combined_scores).all()
