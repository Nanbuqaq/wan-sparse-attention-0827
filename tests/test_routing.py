from __future__ import annotations

import torch

from adapters.routing import (
    _plan_metrics,
    exact_pair_budget_map,
    fixed_edge_budget_map,
    top_p_map,
)
from adapters.routes.stage3 import _tiered_priority


def test_exact_pair_budget_is_deterministic_and_nonempty() -> None:
    scores = torch.tensor(
        [[[[9.0, 1.0, 0.0], [0.0, 8.0, 2.0], [1.0, 0.0, 7.0]]]]
    )
    q_sizes = torch.tensor([[[3, 5, 2]]], dtype=torch.int32)
    k_sizes = torch.tensor([[[4, 1, 5]]], dtype=torch.int32)
    first = exact_pair_budget_map(scores, q_sizes, k_sizes, 0.40)
    second = exact_pair_budget_map(scores, q_sizes, k_sizes, 0.40)
    assert torch.equal(first, second)
    assert torch.all(first.sum(dim=-1) >= 1)
    areas = q_sizes.long().unsqueeze(-1) * k_sizes.long().unsqueeze(-2)
    actual = float((areas * first).sum() / areas.sum())
    assert abs(actual - 0.40) <= 0.12


def test_plan_metrics_accounts_for_padding_and_load() -> None:
    q_sizes = torch.tensor([[[64, 8]]], dtype=torch.int32)
    k_sizes = torch.tensor([[[64, 8]]], dtype=torch.int32)
    block_map = torch.tensor([[[[True, False], [True, True]]]])
    plan = _plan_metrics(
        method="unit",
        backend="fixed64_bf16",
        parameter_origin="test",
        density=0.5,
        block_map=block_map,
        q_sizes=q_sizes,
        k_sizes=k_sizes,
        q_sorted_indices=None,
        k_sorted_indices=None,
        cluster_ms=1.0,
        permutation_ms=2.0,
        selection_ms=3.0,
        metadata={},
    )
    assert plan.logical_pairs == 64 * 64 + 8 * 64 + 8 * 8
    assert plan.scheduled_pairs == 3 * 64 * 64
    assert plan.padding_pairs == plan.scheduled_pairs - plan.logical_pairs
    assert plan.padding_ratio > 0
    assert plan.load_imbalance_max_mean >= 1.0


def test_fixed_edge_budget_has_exact_edge_count() -> None:
    scores = torch.arange(2 * 3 * 5 * 7, dtype=torch.float32).reshape(2, 3, 5, 7)
    mask = fixed_edge_budget_map(scores, 0.2)
    assert torch.all(mask.sum(dim=(-1, -2)) == round(5 * 7 * 0.2))
    assert torch.all(mask.sum(dim=-1) >= 1)


def test_stage3_tiered_priority_preserves_budget_parts() -> None:
    direct = torch.arange(8 * 8, dtype=torch.float32).reshape(1, 1, 8, 8)
    local = -torch.cdist(torch.arange(8).float().view(-1, 1), torch.arange(8).float().view(-1, 1))
    remote = direct.flip(-1)
    remote_mask = local <= -2
    priority, metadata = _tiered_priority(
        direct,
        local,
        remote,
        remote_mask,
        density=0.5,
        base_fraction=0.5,
        local_fraction=0.25,
    )
    selected = fixed_edge_budget_map(priority, 0.5)
    assert torch.all(selected.sum(dim=-1) == 4)
    assert metadata["base_edges_per_row"] == 2
    assert metadata["local_edges_per_row"] == 1
    assert metadata["remote_edges_per_row"] == 1


def test_top_p_map_matches_upstream_weighted_softmax_rule() -> None:
    scores = torch.tensor(
        [[[[2.0, 1.0, -1.0, 0.5], [0.1, 0.2, 0.3, 0.4]]]],
        dtype=torch.float32,
    )
    k_sizes = torch.tensor([[[2, 5, 0, 3]]], dtype=torch.int32)
    actual = top_p_map(scores, k_sizes, top_p=0.70, min_k_ratio=0.25)

    weights = k_sizes.unsqueeze(-2).float()
    logits = scores.float()
    maximum = logits.max(dim=-1, keepdim=True).values
    weighted = weights * torch.exp(logits - maximum)
    weighted = weighted.masked_fill(weights == 0, 0.0)
    probabilities = weighted / weighted.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sorted_probs, order = torch.sort(probabilities, dim=-1, descending=True)
    remove = torch.cumsum(sorted_probs, dim=-1) > 0.70
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    remove[..., :1] = False
    expected = torch.zeros_like(remove)
    expected.scatter_(-1, order, ~remove)
    assert torch.equal(actual, expected)
