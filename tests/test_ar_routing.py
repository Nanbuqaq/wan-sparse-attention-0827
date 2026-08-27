from __future__ import annotations

import pytest
import torch

from adapters.longlive_sparse.ar_routing import route_history
from adapters.longlive_sparse.methods import METHOD_SPECS, validate_method_coverage


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
