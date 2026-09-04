from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path

import pytest
import torch

import adapters.longlive_sparse.selectors as selector_module
from adapters.longlive_sparse import HistoryArchive, SparseHistoryConfig
from adapters.longlive_sparse.rope import apply_selected_rope, build_sparse_positions
from adapters.longlive_sparse.selectors import (
    SparseSelection,
    gather_per_head,
    select_block64_from_tensor,
    summarize_query_for_pretransfer,
)
from adapters.longlive_sparse.stats import SparseCallRecord, SparseRunStats, TimingBreakdown
from adapters.longlive_sparse.transfer_plan import build_transfer_plan


def _frame(frame_id: int, *, tokens: int = 8, heads: int = 2, dim: int = 12):
    base = torch.arange(tokens * heads * dim, dtype=torch.float32).reshape(
        1, tokens, heads, dim
    )
    key = base + frame_id * 1000
    value = base * 0.1 + frame_id * 100
    return key, value


@pytest.mark.parametrize("method", ["block64_history", "kcluster32_history"])
def test_exact_budget_is_deterministic(method: str):
    config = SparseHistoryConfig(
        method=method,
        history_density=0.25,
        block_size=3,
        clusters_per_frame=2,
        kmeans_iterations=4,
    )
    archive = HistoryArchive(config, spatial_height=2, spatial_width=4)
    for frame_id in (4, 8):
        key, value = _frame(frame_id)
        archive.index_frame(0, frame_id, key, value)
    query = torch.randn(1, 6, 2, 12, generator=torch.Generator().manual_seed(7))
    first = archive.select(0, query, [4, 8])
    second = archive.select(0, query, [4, 8])
    assert first.candidate_history_tokens == 16
    assert first.selected_history_tokens == 4
    assert first.frame_ids.shape == (1, 2, 4)
    assert torch.equal(first.frame_ids, second.frame_ids)
    assert torch.equal(first.token_ids, second.token_ids)
    for head in range(2):
        coordinates = list(
            zip(first.frame_ids[0, head].tolist(), first.token_ids[0, head].tolist())
        )
        assert coordinates == sorted(coordinates)
        assert len(coordinates) == len(set(coordinates))


@pytest.mark.parametrize("method", ["block64_history", "kcluster32_history"])
def test_full_density_restores_original_kv_order(method: str):
    config = SparseHistoryConfig(
        method=method,
        history_density=1.0,
        block_size=3,
        clusters_per_frame=2,
    )
    archive = HistoryArchive(config, spatial_height=2, spatial_width=4)
    expected_k = []
    expected_v = []
    for frame_id in (2, 5):
        key, value = _frame(frame_id)
        expected_k.append(key)
        expected_v.append(value)
        archive.index_frame(0, frame_id, key, value)
    query = torch.randn(1, 4, 2, 12)
    selection = archive.select(0, query, [2, 5])
    materialized = archive.materialize(
        0,
        selection,
        device="cpu",
        current_frame_id=9,
        freqs=None,
        candidate_frame_ids=[2, 5],
    )
    assert torch.equal(materialized.key, torch.cat(expected_k, dim=1))
    assert torch.equal(materialized.value, torch.cat(expected_v, dim=1))
    assert materialized.transferred_bytes == (
        materialized.key.numel() + materialized.value.numel()
    ) * materialized.key.element_size()


def test_vectorized_dense_materialization_handles_unsorted_candidate_frames():
    config = SparseHistoryConfig(method="block64_history", history_density=1.0)
    archive = HistoryArchive(config, spatial_height=2, spatial_width=4)
    key2, value2 = _frame(2)
    key5, value5 = _frame(5)
    archive.index_frame(0, 2, key2, value2)
    archive.index_frame(0, 5, key5, value5)
    selection = archive.select(0, torch.randn(1, 4, 2, 12), [2, 5])
    dense_key = torch.cat((key5, key2), dim=1)
    dense_value = torch.cat((value5, value2), dim=1)
    dense_frames = torch.tensor([5] * 8 + [2] * 8).view(1, 1, 16).expand(1, 2, -1)
    dense_tokens = torch.arange(8).repeat(2).view(1, 1, 16).expand(1, 2, -1)
    materialized = archive.materialize(
        0,
        selection,
        device="cpu",
        current_frame_id=9,
        freqs=None,
        candidate_frame_ids=[5, 2],
        dense_key=dense_key,
        dense_value=dense_value,
        dense_frame_ids=dense_frames,
        dense_token_ids=dense_tokens,
    )
    assert torch.equal(materialized.key, torch.cat((key2, key5), dim=1))
    assert torch.equal(materialized.value, torch.cat((value2, value5), dim=1))


def test_transfer_plan_materialization_preserves_logical_kv_and_reports_padding():
    config = SparseHistoryConfig(method="block64_history", history_density=0.25)
    archive = HistoryArchive(config, spatial_height=2, spatial_width=4)
    key2, value2 = _frame(2)
    key5, value5 = _frame(5)
    archive.index_frame(0, 2, key2, value2)
    archive.index_frame(0, 5, key5, value5)
    route = archive.route_indexed(
        0, torch.randn(1, 4, 2, 12), [2, 5], exact_k_tokens=8
    )
    selection = SparseSelection(
        frame_ids=route.union_frame_ids,
        token_ids=route.union_token_ids,
        scores=torch.zeros_like(route.union_frame_ids, dtype=torch.float32),
        candidate_history_tokens=route.candidate_history_tokens,
        selected_history_tokens=route.unique_history_tokens,
        candidate_units=route.candidate_history_tokens,
        selected_units=route.unique_history_tokens,
        cluster_size_min=None,
        cluster_size_max=None,
        index_bytes=0,
        timing=TimingBreakdown(),
    )
    legacy = archive.materialize(
        0,
        selection,
        device="cpu",
        current_frame_id=9,
        freqs=None,
        candidate_frame_ids=[2, 5],
    )
    exact_plan = build_transfer_plan(
        route,
        [2, 5],
        frame_tokens=8,
        layout="exact_compact",
        bytes_per_token=2 * 12 * key2.element_size(),
    )
    exact = archive.materialize_transfer_plan(
        0,
        exact_plan,
        route,
        device="cpu",
        current_frame_id=9,
        freqs=None,
    )
    assert torch.equal(exact.key, legacy.key)
    assert torch.equal(exact.value, legacy.value)
    assert exact.transfer_plan_sha256 == exact_plan.digest()
    block_plan = build_transfer_plan(
        route,
        [2, 5],
        frame_tokens=8,
        layout="block64",
        bytes_per_token=2 * 12 * key2.element_size(),
    )
    blocked = archive.materialize_transfer_plan(
        0,
        block_plan,
        route,
        device="cpu",
        current_frame_id=9,
        freqs=None,
    )
    assert torch.equal(blocked.key, legacy.key)
    assert blocked.transferred_bytes > exact.transferred_bytes
    assert blocked.padding_bytes > 0


@pytest.mark.parametrize(
    "method",
    [
        "block64_history",
        "kcluster32_history",
        "fixed_k128_history",
        "fixed_k256_history",
        "qlocal_kmeans8_ar",
        "radius_k256_ar",
        "temporal_k256_t16_ar",
    ],
)
def test_indexed_pretransfer_routes_are_exact_budget(method: str):
    config = SparseHistoryConfig(
        method=method,
        history_density=0.25,
        block_size=4,
        clusters_per_frame=2,
        kmeans_iterations=2,
        method_params={"iterations": 2} if method != "block64_history" else {},
    )
    archive = HistoryArchive(config, spatial_height=2, spatial_width=4)
    for frame_id in (5, 2):
        key, value = _frame(frame_id)
        archive.index_frame(0, frame_id, key, value)
    plan = archive.route_indexed(
        0,
        torch.randn(1, 16, 2, 12, generator=torch.Generator().manual_seed(9)),
        [5, 2],
        exact_k_tokens=8,
    )
    assert plan.history_pair_density == pytest.approx(0.25)
    assert plan.routing_stage == "pre-transfer"
    assert plan.metadata["index_source"] == "per_frame_archive"


def test_indexed_full_density_fast_path_routes_all_history():
    config = SparseHistoryConfig(method="block64_history", history_density=1.0)
    archive = HistoryArchive(config, spatial_height=2, spatial_width=4)
    key, value = _frame(4)
    archive.index_frame(0, 4, key, value)
    plan = archive.route_indexed(
        0, torch.randn(1, 16, 2, 12), [4], exact_k_tokens=8
    )
    assert plan.history_pair_density == 1.0
    assert plan.history_transfer_density == 1.0
    assert plan.metadata["full_density_fast_path"] is True


@pytest.mark.parametrize(
    "method",
    [
        "coverage_cluster_history",
        "vaware_cluster_history",
        "transfer_vaware_hybrid_history",
    ],
)
def test_proposed_routes_use_only_query_summaries_and_cpu_kv_prototypes(
    method: str,
):
    params = {
        "remote_min_frames": 1,
    }
    if method == "transfer_vaware_hybrid_history":
        params["transfer_multiplier"] = 1.25
    config = SparseHistoryConfig(
        method=method,
        history_density=0.5,
        block_size=4,
        kmeans_iterations=2,
        method_params=params,
    )
    archive = HistoryArchive(config, spatial_height=2, spatial_width=4)
    for frame_id in (5, 2):
        key, value = _frame(frame_id)
        archive.index_frame(0, frame_id, key, value)
    query = torch.randn(1, 16, 2, 12, generator=torch.Generator().manual_seed(12))
    summary = summarize_query_for_pretransfer(query, block_size=4)
    plan = archive.route_indexed(0, summary, [5, 2], exact_k_tokens=8)
    assert plan.history_pair_density == pytest.approx(0.5)
    assert plan.metadata["query_summary_only"] is True
    assert plan.metadata["query_summary_bytes"] == summary.summary_bytes
    assert plan.metadata["query_summary_block_size"] == 4
    assert plan.metadata["output_residual_online"] is False
    assert plan.metadata["output_residual_role"] == "offline_teacher_only"
    assert plan.metadata["executes_original_kv"] is True
    assert plan.metadata["online_proxy"].startswith("q_summary")
    assert plan.metadata["remote_prototype_policy"] == "block64_kv_mean"
    assert plan.metadata["remote_prototypes_per_frame"] == 2
    if method == "transfer_vaware_hybrid_history":
        assert plan.history_transfer_density <= 0.625 + 1e-6


def test_proposed_index_uses_block_means_without_token_kmeans(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("proposed Block64 prototype path called token K-means")

    monkeypatch.setattr(
        selector_module, "_batched_spherical_kmeans", fail_if_called
    )
    config = SparseHistoryConfig(
        method="vaware_cluster_history",
        history_density=0.5,
        block_size=4,
        method_params={"remote_min_frames": 1},
    )
    archive = HistoryArchive(config, spatial_height=2, spatial_width=4)
    key, value = _frame(3)
    index = archive.index_frame(0, 3, key, value)
    assert index.block_centroids.shape == (1, 2, 2, 12)
    assert index.block_value_centroids.shape == (1, 2, 2, 12)
    assert index.cluster_centroids.numel() == 0
    assert index.block_cluster_membership.numel() == 0
    assert index.routing_bytes > 0


@pytest.mark.parametrize("use_allowed_mask", [False, True])
def test_tensorized_tier_expansion_matches_token_loop_reference(use_allowed_mask):
    entries = [(0, 0, 4), (0, 4, 7), (1, 0, 4), (1, 4, 7)]
    frame_tokens = 7
    orders = ([2, 0, 3, 1], [0, 1, 2, 3], [3, 1, 2, 0])
    tier_counts = (5, 3, 2)
    allowed_set = set(range(14)) - {1, 5, 9} if use_allowed_mask else None

    selected = []
    selected_set = set()

    def add_until(order, target):
        for unit in order:
            frame_index, start, end = entries[unit]
            for token in range(start, end):
                flat = frame_index * frame_tokens + token
                if flat in selected_set:
                    continue
                if allowed_set is not None and flat not in allowed_set:
                    continue
                selected.append(flat)
                selected_set.add(flat)
                if len(selected) == target:
                    return

    base, local, remote = tier_counts
    add_until(orders[0], base)
    add_until(orders[1], base + local)
    add_until(orders[2], base + local + remote)
    if len(selected) < sum(tier_counts):
        add_until(list(dict.fromkeys((*orders[0], *orders[1], *orders[2]))), 10)

    block_tokens = selector_module._block_token_table(entries, frame_tokens)
    allowed_mask = None
    if allowed_set is not None:
        allowed_mask = torch.zeros(14, dtype=torch.bool)
        allowed_mask[list(allowed_set)] = True
    actual = selector_module._expand_tiered_block_orders(
        block_tokens,
        tuple(torch.tensor(order) for order in orders),
        tier_counts,
        allowed_tokens=allowed_mask,
    )
    assert torch.equal(actual, torch.tensor(sorted(selected)))


def test_proposed_config_rejects_removed_token_cluster_parameter():
    with pytest.raises(ValueError, match="unknown method_params fields"):
        SparseHistoryConfig(
            method="coverage_cluster_history",
            method_params={"remote_clusters": 25},
        )


def test_per_head_gather_supports_different_token_sets():
    tensor = torch.arange(1 * 7 * 2 * 3).reshape(1, 7, 2, 3)
    indices = torch.tensor([[[0, 2, 6], [1, 3, 5]]])
    gathered = gather_per_head(tensor, indices)
    assert gathered.shape == (1, 3, 2, 3)
    assert torch.equal(gathered[0, :, 0], tensor[0, [0, 2, 6], 0])
    assert torch.equal(gathered[0, :, 1], tensor[0, [1, 3, 5], 1])


def test_native_block_selector_is_exact_and_full_density_is_identity():
    query = torch.randn(1, 5, 2, 12, generator=torch.Generator().manual_seed(3))
    key = torch.randn(1, 11, 2, 12, generator=torch.Generator().manual_seed(4))
    full = select_block64_from_tensor(query, key, 1.0, 4)
    assert torch.equal(full, torch.arange(11).view(1, 1, 11).expand(1, 2, 11))
    sparse = select_block64_from_tensor(query, key, 0.25, 4)
    assert sparse.shape == (1, 2, 3)
    assert torch.all(sparse[:, :, 1:] >= sparse[:, :, :-1])


def _freqs(max_position: int, complex_dim: int) -> torch.Tensor:
    positions = torch.arange(max_position, dtype=torch.float64).unsqueeze(1)
    rates = torch.linspace(0.01, 0.2, complex_dim, dtype=torch.float64).unsqueeze(0)
    return torch.polar(torch.ones_like(positions * rates), positions * rates)


def test_sparse_rope_matches_direct_per_token_reference():
    key = torch.randn(1, 4, 2, 12, dtype=torch.float32)
    frame_ids = torch.tensor([[[4, 4, 7, 7], [4, 7, 7, 9]]])
    token_ids = torch.tensor([[[0, 3, 4, 7], [1, 2, 5, 6]]])
    positions = build_sparse_positions(
        frame_ids=frame_ids,
        token_ids=token_ids,
        current_frame_id=10,
        spatial_width=4,
        rope_policy="clipped_relative_age",
        max_relative_age=8,
    )
    freqs = _freqs(16, 6)
    actual = apply_selected_rope(key, positions, freqs)

    split = [2, 2, 2]
    ft, fh, fw = freqs.split(split, dim=1)
    expected = torch.empty_like(key)
    for batch in range(1):
        for token in range(4):
            for head in range(2):
                t, y, x = positions[batch, head, token].tolist()
                rotation = torch.cat((ft[t], fh[y], fw[x]))
                source = torch.view_as_complex(
                    key[batch, token, head].double().reshape(-1, 2)
                )
                expected[batch, token, head] = torch.view_as_real(
                    source * rotation
                ).flatten().float()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_rope_position_policies_are_explicit():
    frame_ids = torch.tensor([[[3, 7, 9]]])
    token_ids = torch.tensor([[[0, 5, 7]]])
    zero = build_sparse_positions(
        frame_ids=frame_ids,
        token_ids=token_ids,
        current_frame_id=12,
        spatial_width=4,
        rope_policy="upstream_zero",
        max_relative_age=5,
    )
    rank = build_sparse_positions(
        frame_ids=frame_ids,
        token_ids=token_ids,
        current_frame_id=12,
        spatial_width=4,
        rope_policy="recency_rank",
        max_relative_age=5,
        candidate_frame_ids=torch.tensor([9, 3, 7]),
    )
    age = build_sparse_positions(
        frame_ids=frame_ids,
        token_ids=token_ids,
        current_frame_id=12,
        spatial_width=4,
        rope_policy="clipped_relative_age",
        max_relative_age=5,
    )
    assert zero[..., 0].tolist() == [[[0, 0, 0]]]
    assert rank[..., 0].tolist() == [[[0, 1, 2]]]
    assert age[..., 0].tolist() == [[[5, 5, 3]]]


def test_run_stats_reports_history_and_global_density_separately():
    stats = SparseRunStats(method="block64_history")
    record = SparseCallRecord(
        layer_id=0,
        method="block64_history",
        candidate_frames=2,
        candidate_history_tokens=100,
        selected_history_tokens=25,
        exact_tokens=75,
        query_tokens=10,
        dense_k_tokens=175,
        executed_k_tokens=100,
        transferred_bytes=400,
        index_bytes=20,
        query_summary_bytes=96,
        candidate_transfer_bytes=1600,
        full_history_pairs=2000,
        selected_history_pairs=500,
        dense_qk_pairs_value=3500,
        executed_qk_pairs_value=2000,
        route_plan_sha256="a" * 64,
    )
    stats.record_call(record)
    assert math.isclose(stats.history_density, 0.25)
    assert math.isclose(stats.global_executed_density, 100 / 175)
    payload = stats.as_dict()
    assert payload["history_density"] == pytest.approx(0.25)
    assert payload["history_transfer_density"] == pytest.approx(0.25)
    assert payload["global_executed_density"] == pytest.approx(100 / 175)
    assert payload["query_summary_bytes"] == 96
    assert payload["route_plan_sha256_counts"] == {"a" * 64: 1}


def test_pinned_source_provenance_hashes_are_current():
    root = Path(__file__).resolve().parents[1]
    provenance_path = root / "results/manifests/source_provenance.json"
    if not provenance_path.is_file():
        pytest.skip("local source-provenance manifest is not included in the public branch")
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    for source in payload.values():
        base = Path(source["path"])
        for relative, expected in source["source_hashes"].items():
            digest = hashlib.sha256((base / relative).read_bytes()).hexdigest()
            assert digest == expected


def test_run_stats_merge_preserves_density_denominators():
    first = SparseRunStats(method="block64_history")
    second = SparseRunStats(method="block64_history")
    first.record_call(
        SparseCallRecord(
            layer_id=0,
            method="block64_history",
            candidate_frames=1,
            candidate_history_tokens=100,
            selected_history_tokens=25,
            exact_tokens=50,
            query_tokens=2,
            dense_k_tokens=150,
            executed_k_tokens=75,
            transferred_bytes=100,
            index_bytes=10,
        )
    )
    second.record_call(
        SparseCallRecord(
            layer_id=0,
            method="block64_history",
            candidate_frames=1,
            candidate_history_tokens=200,
            selected_history_tokens=100,
            exact_tokens=50,
            query_tokens=2,
            dense_k_tokens=250,
            executed_k_tokens=150,
            transferred_bytes=200,
            index_bytes=20,
        )
    )
    first.merge(second)
    assert first.history_density == pytest.approx(125 / 300)
    assert first.global_executed_density == pytest.approx((2 * 75 + 2 * 150) / (2 * 150 + 2 * 250))
    assert first.index_transfer_bytes == 30
