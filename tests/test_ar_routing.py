from __future__ import annotations

import pytest
import torch

from adapters.longlive_sparse.ar_routing import route_history
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.methods import METHOD_SPECS, validate_method_coverage
from adapters.longlive_sparse.runtime_attention import SparseHistorySelfAttention


SELF_METHODS = [
    "qlocal_kmeans8_ar",
    "radius_k256_ar",
    "qmetric_k256_r32_ar",
    "temporal_k256_t16_ar",
    "sizesplit_k128_c2_ar",
]
PAPER_METHODS = ["svg2_ar", "adacluster_ar", "svoo_ar", "scope_ar"]


def inputs():
    generator = torch.Generator().manual_seed(20260827)
    query = torch.randn(1, 32, 2, 128, generator=generator)
    key = torch.randn(1, 48, 2, 128, generator=generator)
    frame_ids = torch.arange(3).repeat_interleave(16).view(1, 1, 48).expand(1, 2, -1)
    token_ids = torch.arange(16).repeat(3).view(1, 1, 48).expand(1, 2, -1)
    return query, key, frame_ids, token_ids


def test_method_categories_do_not_mix_baselines_with_self_clusters():
    validate_method_coverage()
    assert sum(spec.counts_as_self_cluster for spec in METHOD_SPECS.values()) == 5
    assert all(METHOD_SPECS[name].category == "self_cluster" for name in SELF_METHODS)
    assert all(METHOD_SPECS[name].category == "paper" for name in PAPER_METHODS)
    assert not METHOD_SPECS["kcluster32_history"].counts_as_self_cluster
    assert METHOD_SPECS["native_dense"].routing_stage == "N/A"
    assert METHOD_SPECS["rag_dense"].routing_stage == "post-transfer"


def test_profiled_rag_dense_routes_all_history():
    query, key, frame_ids, token_ids = inputs()
    plan = route_history(
        query,
        key,
        frame_ids,
        token_ids,
        method="rag_dense",
        density=1.0,
        exact_k_tokens=16,
    )
    assert plan.history_pair_density == 1.0
    assert plan.history_transfer_density == 1.0
    assert plan.global_executed_density == 1.0


@pytest.mark.parametrize(
    "method",
    [
        "random_history",
        "block64_history",
        "token_oracle",
        "kcluster32_history",
        "fixed_k128_history",
        *SELF_METHODS,
        *PAPER_METHODS,
    ],
)
def test_all_routes_produce_exact_accountable_plan(method: str):
    query, key, frame_ids, token_ids = inputs()
    plan = route_history(
        query,
        key,
        frame_ids,
        token_ids,
        method=method,
        density=0.25,
        exact_k_tokens=16,
    )
    assert plan.method == method
    assert plan.query_labels.shape == (1, 2, 32)
    assert plan.history_pair_density == pytest.approx(0.25, abs=0.08)
    assert 0.0 <= plan.history_transfer_density <= 1.0
    assert 0.0 < plan.global_executed_density <= 1.0
    assert len(plan.digest()) == 64
    for head in range(2):
        groups = int(plan.query_labels[0, head].max()) + 1
        for group in range(groups):
            count = int(plan.group_history_counts[0, head, group])
            indices = plan.group_union_indices[0, head, group, :count]
            assert (indices >= 0).all()
            assert (indices < plan.union_frame_ids.shape[-1]).all()


def test_rag_local_has_zero_history_pairs_and_transfer():
    query, key, frame_ids, token_ids = inputs()
    plan = route_history(
        query,
        key,
        frame_ids,
        token_ids,
        method="rag_local",
        density=0.25,
        exact_k_tokens=16,
    )
    assert plan.history_pairs == 0
    assert plan.unique_history_tokens == 0
    assert plan.history_pair_density == 0
    assert plan.history_transfer_density == 0


def test_post_transfer_union_coordinates_map_back_to_dense_candidate_order():
    query, key, frame_ids, token_ids = inputs()
    permutation = torch.cat((torch.arange(32, 48), torch.arange(0, 32)))
    key = key.index_select(1, permutation)
    frame_ids = frame_ids.index_select(2, permutation)
    token_ids = token_ids.index_select(2, permutation)
    plan = route_history(
        query,
        key,
        frame_ids,
        token_ids,
        method="token_oracle",
        density=0.25,
        exact_k_tokens=16,
    )
    indices = SparseHistorySelfAttention._union_indices_from_coordinates(
        plan, frame_ids, token_ids
    )
    gathered_frames = frame_ids.gather(-1, indices)
    gathered_tokens = token_ids.gather(-1, indices)
    valid = plan.union_frame_ids >= 0
    assert torch.equal(gathered_frames[valid], plan.union_frame_ids[valid])
    assert torch.equal(gathered_tokens[valid], plan.union_token_ids[valid])


def test_method_parameter_overrides_are_explicit_and_identity_safe():
    config = SparseHistoryConfig(
        method="svg2_ar",
        method_params={"q_clusters": 64, "k_clusters": 128, "iterations": 2},
    )
    assert config.method_params["k_clusters"] == 128
    with pytest.raises(ValueError, match="cannot change method identity"):
        SparseHistoryConfig(
            method="svg2_ar", method_params={"routing_stage": "pre-transfer"}
        )
