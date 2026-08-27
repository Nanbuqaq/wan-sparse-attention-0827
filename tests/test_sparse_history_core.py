from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path

import pytest
import torch

from adapters.longlive_sparse import HistoryArchive, SparseHistoryConfig
from adapters.longlive_sparse.rope import apply_selected_rope, build_sparse_positions
from adapters.longlive_sparse.selectors import (
    gather_per_head,
    select_block64_from_tensor,
)
from adapters.longlive_sparse.stats import SparseCallRecord, SparseRunStats


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
    )
    stats.record_call(record)
    assert math.isclose(stats.history_density, 0.25)
    assert math.isclose(stats.global_executed_density, 100 / 175)
    payload = stats.as_dict()
    assert payload["history_density"] == pytest.approx(0.25)
    assert payload["global_executed_density"] == pytest.approx(100 / 175)


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
