from __future__ import annotations

import torch

from adapters.longlive_sparse.ar_routing import route_history
from adapters.longlive_sparse.attention_bias import AttentionBiasPlan
from adapters.longlive_sparse.backends import (
    execute_biased_sdpa_reference,
    execute_grouped_fa2,
    execute_kvout_online_reference,
)
from adapters.longlive_sparse.route_plan import HistoryRoutePlan


def inputs():
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(1, 16, 2, 32, generator=generator)
    history_key = torch.randn(1, 24, 2, 32, generator=generator)
    history_value = torch.randn(1, 24, 2, 32, generator=generator)
    exact_key = torch.randn(1, 8, 2, 32, generator=generator)
    exact_value = torch.randn(1, 8, 2, 32, generator=generator)
    frame_ids = torch.arange(3).repeat_interleave(8).view(1, 1, 24).expand(1, 2, -1)
    token_ids = torch.arange(8).repeat(3).view(1, 1, 24).expand(1, 2, -1)
    return query, exact_key, exact_value, history_key, history_value, frame_ids, token_ids


def test_grouped_backend_replays_same_route_plan_on_cpu():
    query, exact_key, exact_value, history_key, history_value, frame_ids, token_ids = inputs()
    plan = route_history(
        query,
        history_key,
        frame_ids,
        token_ids,
        method="block64_history",
        density=0.25,
        exact_k_tokens=8,
    )
    first = execute_grouped_fa2(query, exact_key, exact_value, history_key, history_value, plan)
    second = execute_grouped_fa2(query, exact_key, exact_value, history_key, history_value, plan)
    torch.testing.assert_close(first.output, second.output, atol=0, rtol=0)
    assert first.route_plan_sha256 == plan.digest() == second.route_plan_sha256
    assert first.logical_pairs == second.logical_pairs


def test_route_plan_state_roundtrip_preserves_sha():
    query, exact_key, exact_value, history_key, history_value, frame_ids, token_ids = inputs()
    plan = route_history(
        query,
        history_key,
        frame_ids,
        token_ids,
        method="block64_history",
        density=0.5,
        exact_k_tokens=exact_key.shape[1],
    )
    restored = HistoryRoutePlan.from_state_dict(plan.state_dict())
    assert restored.digest() == plan.digest()
    assert restored.as_dict() == plan.as_dict()


def test_grouped_backend_executes_exact_only_for_zero_history_route():
    query, exact_key, exact_value, history_key, history_value, frame_ids, token_ids = inputs()
    plan = route_history(
        query,
        history_key,
        frame_ids,
        token_ids,
        method="rag_local",
        density=0.25,
        exact_k_tokens=exact_key.shape[1],
    )
    result = execute_grouped_fa2(
        query,
        exact_key,
        exact_value,
        history_key[:, :0],
        history_value[:, :0],
        plan,
    )
    assert result.output.shape == query.shape
    assert result.logical_pairs == query.shape[0] * query.shape[2] * query.shape[1] * exact_key.shape[1]


def test_kvout_online_reference_matches_grouped_sdpa_reference():
    query, exact_key, exact_value, history_key, history_value, frame_ids, token_ids = inputs()
    plan = route_history(
        query,
        history_key,
        frame_ids,
        token_ids,
        method="block64_history",
        density=0.25,
        exact_k_tokens=8,
    )
    grouped = execute_grouped_fa2(
        query, exact_key, exact_value, history_key, history_value, plan
    )
    kvout = execute_kvout_online_reference(
        query, exact_key, exact_value, history_key, history_value, plan, block_tokens=5
    )
    torch.testing.assert_close(kvout.output, grouped.output, atol=2e-5, rtol=2e-5)
    assert kvout.route_plan_sha256 == grouped.route_plan_sha256 == plan.digest()


def test_biased_sdpa_reference_uses_compact_role_plan():
    query, exact_key, exact_value, history_key, history_value, frame_ids, token_ids = inputs()
    plan = route_history(
        query,
        history_key,
        frame_ids,
        token_ids,
        method="rag_dense",
        density=1.0,
        exact_k_tokens=8,
    )
    union_width = plan.union_frame_ids.shape[-1]
    query_roles = torch.zeros(1, query.shape[1], 2)
    query_roles[..., 0] = 1.0
    history_roles = torch.zeros(1, query.shape[2], union_width, 2)
    history_roles[..., 0] = 1.0
    bias = AttentionBiasPlan(
        role_names=("identity", "scene"),
        query_role_probabilities=query_roles,
        history_role_probabilities=history_roles,
        history_age_weights=torch.ones(1, query.shape[2], union_width),
        mode="oracle_test",
        metadata={"context_weight": 0.2},
    )
    biased = execute_biased_sdpa_reference(
        query,
        exact_key,
        exact_value,
        history_key,
        history_value,
        plan,
        bias,
    )
    grouped = execute_grouped_fa2(
        query, exact_key, exact_value, history_key, history_value, plan
    )
    torch.testing.assert_close(biased.output, grouped.output, atol=2e-5, rtol=2e-5)
