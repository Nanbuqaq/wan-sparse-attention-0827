from __future__ import annotations

import pytest
import torch

from adapters.longlive_sparse import HistoryArchive, SparseHistoryConfig
from adapters.longlive_sparse.history_cache import (
    CachedHistoryKV,
    HistoryKVCacheKey,
    HistoryUnionCache,
    tensor_sha256,
)
from adapters.longlive_sparse.reuse import RouteReuseTracker, set_jaccard
from adapters.longlive_sparse.route_plan import HistoryRoutePlan
from adapters.longlive_sparse.timeline import (
    TimelineInterval,
    interval_union_duration,
    overlap_duration,
)


def _key(**overrides) -> HistoryKVCacheKey:
    values = {
        "layer_id": 0,
        "archive_epoch": 1,
        "storage_version": 4,
        "current_frame_id": 12,
        "candidate_frame_ids": (1, 2, 3),
        "selected_coordinate_sha256": "a" * 64,
        "route_plan_sha256": "b" * 64,
        "rope_policy": "upstream_zero",
        "rope_position_sha256": "c" * 64,
        "dtype": "torch.bfloat16",
        "device": "cuda:0",
        "transfer_layout": "exact_compact",
        "padding_strategy": "rectangular_head_max",
    }
    values.update(overrides)
    return HistoryKVCacheKey(**values)


def _entry(key: HistoryKVCacheKey, tokens: int = 2) -> CachedHistoryKV:
    tensor = torch.zeros(1, tokens, 1, 4)
    return CachedHistoryKV(
        key=key,
        value=tensor.clone(),
        key_unrotated=tensor.clone(),
        key_roped=tensor.clone(),
        positions=torch.zeros(1, 1, tokens, 3, dtype=torch.long),
    )


def _plan(token: int) -> HistoryRoutePlan:
    return HistoryRoutePlan(
        method="block64_history",
        routing_stage="pre-transfer",
        query_labels=torch.tensor([[[0, 0]]]),
        query_group_sizes=torch.tensor([[[2]]]),
        union_frame_ids=torch.tensor([[[1]]]),
        union_token_ids=torch.tensor([[[token]]]),
        group_union_indices=torch.tensor([[[[0]]]]),
        group_history_counts=torch.tensor([[[1]]]),
        candidate_history_tokens=4,
        query_tokens=2,
        exact_k_tokens=2,
        target_history_density=0.25,
    )


def test_cache_key_changes_for_every_semantic_dimension() -> None:
    baseline = _key().digest()
    assert _key(storage_version=5).digest() != baseline
    assert _key(current_frame_id=15).digest() != baseline
    assert _key(rope_policy="recency_rank").digest() != baseline
    assert _key(transfer_layout="block64").digest() != baseline


def test_cache_is_budgeted_and_records_hits_misses_evictions() -> None:
    first = _entry(_key())
    cache = HistoryUnionCache(first.bytes + 1)
    assert cache.get(first.key) is None
    cache.put(first)
    assert cache.get(first.key) is first
    second = _entry(_key(current_frame_id=15))
    cache.put(second)
    assert cache.evictions == 1
    assert cache.get(first.key) is None
    assert cache.get(second.key) is second


def test_cache_rejects_entry_larger_than_explicit_budget() -> None:
    entry = _entry(_key())
    with pytest.raises(MemoryError, match="exceeds"):
        HistoryUnionCache(entry.bytes - 1).put(entry)


def test_archive_epoch_and_storage_version_are_monotonic() -> None:
    archive = HistoryArchive(
        SparseHistoryConfig(method="block64_history"),
        spatial_height=1,
        spatial_width=2,
    )
    epoch = archive.epoch
    key = torch.zeros(1, 2, 1, 4)
    archive.index_frame(0, 1, key, key.clone())
    assert archive.storage_version == 1
    archive.clear_frames()
    assert archive.epoch == epoch + 1
    assert archive.storage_version == 0


def test_route_reuse_tracker_reports_denoising_jaccard() -> None:
    tracker = RouteReuseTracker()
    first = _plan(0)
    second = _plan(1)
    tracker.record(first, layer_id=0, chunk_id=4, denoising_pass=0)
    tracker.record(first, layer_id=0, chunk_id=4, denoising_pass=1)
    tracker.record(second, layer_id=0, chunk_id=4, denoising_pass=2)
    summary = tracker.denoising_summary(layer_id=0, chunk_id=4)
    assert summary["passes"] == 3
    assert summary["records"][1]["same_route_sha_as_first"] is True
    assert summary["records"][2]["jaccard_vs_first"] == 0.0
    assert set_jaccard([], []) == 1.0
    assert len(tensor_sha256(torch.zeros(2))) == 64


def test_timeline_reports_union_and_true_overlap() -> None:
    copy = [TimelineInterval("h2d", 0.0, 3.0, "copy")]
    compute = [
        TimelineInterval("attn0", 1.0, 2.0, "compute"),
        TimelineInterval("attn1", 2.5, 4.0, "compute"),
    ]
    assert overlap_duration(copy, compute) == pytest.approx(1.5)
    assert interval_union_duration(copy + compute) == pytest.approx(4.0)
