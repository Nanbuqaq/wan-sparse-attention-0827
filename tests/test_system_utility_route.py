from __future__ import annotations

import torch

from adapters.longlive_sparse.contexts import OnlineRoutingContext
from adapters.longlive_sparse.cost_model import HardwareCostProfile, SystemCostModel
from adapters.longlive_sparse.system_utility_route import (
    SystemUtilityRouteConfig,
    build_cost_model_set_cost_factory,
    build_system_utility_route,
)


def _context() -> OnlineRoutingContext:
    generator = torch.Generator().manual_seed(12)
    frames = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    starts = torch.tensor([0, 4, 0, 4, 0, 4, 0, 4])
    return OnlineRoutingContext(
        query_centroids=torch.randn(1, 2, 4, 4, generator=generator),
        query_group_sizes=torch.full((1, 2, 4), 2, dtype=torch.long),
        key_prototypes=torch.randn(1, 2, 8, 4, generator=generator),
        value_prototypes=torch.randn(1, 2, 8, 4, generator=generator),
        block_frame_ids=frames,
        block_token_starts=starts,
        block_token_ends=starts + 4,
        block_age=3 - frames,
        metadata={"candidate_frame_ids": [0, 1, 2, 3], "layer_id": 0},
    )


def test_system_utility_route_is_causal_bounded_and_changes_sha() -> None:
    context = _context()
    legacy = build_system_utility_route(
        context,
        exact_k_tokens=8,
        config=SystemUtilityRouteConfig(
            value_candidate="sum_value",
            cost_strategy="static_block",
            history_density=0.25,
            correlation_fraction=0.70,
            coverage_fraction=0.15,
            remote_fraction=0.15,
        ),
    )
    top_p = build_system_utility_route(
        context,
        exact_k_tokens=8,
        config=SystemUtilityRouteConfig(
            value_candidate="sum_value",
            cost_strategy="static_block",
            history_density=0.25,
            correlation_fraction=0.70,
            coverage_fraction=0.15,
            remote_fraction=0.15,
            group_selection_policy="mass_preserving_top_p",
            group_top_p=0.5,
            group_min_k_ratio=0.0,
        ),
    )
    cap = legacy.metadata["physical_union_cap_tokens"]
    assert max(legacy.metadata["selected_union_tokens"][0]) <= cap
    assert legacy.method == "system_utility_history"
    assert legacy.metadata["online_context_only"] is True
    assert legacy.metadata["offline_teacher_used"] is False
    assert top_p.history_pairs <= legacy.history_pairs
    assert top_p.digest() != legacy.digest()


def test_system_utility_route_can_use_frozen_marginal_cost_model() -> None:
    context = _context()
    profile = HardwareCostProfile(
        profile_id="test-hardware",
        model_version="test-v1",
        h2d_bytes_per_second=1e10,
        hbm_bytes_per_second=1e12,
        copy_launch_seconds=1e-5,
        pack_run_seconds=2e-5,
        pack_bytes_per_second=2e10,
        source_artifact_sha256="a" * 64,
    )
    factory = build_cost_model_set_cost_factory(
        context,
        SystemCostModel(profile),
        exact_k_tokens=8,
        transfer_layout="block64",
        transfer_mode="packed_separate",
        execution_dataflow="qout_grouped_fa2",
    )
    route = build_system_utility_route(
        context,
        exact_k_tokens=8,
        config=SystemUtilityRouteConfig(
            value_candidate="peak_value",
            cost_strategy="marginal_set",
            history_density=0.25,
            correlation_fraction=1.0,
            coverage_fraction=0.0,
            remote_fraction=0.0,
        ),
        set_cost_factory=factory,
    )
    assert route.unique_history_tokens <= 16
    assert route.metadata["route_config"]["cost_strategy"] == "marginal_set"
