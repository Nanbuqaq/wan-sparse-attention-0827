from dataclasses import replace

import pytest
import torch

from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.history_cache import RawHistoryBlockCache
from adapters.longlive_sparse.raw_composition import compile_raw_requests
from adapters.longlive_sparse.route_plan import HistoryRoutePlan
from adapters.longlive_sparse.staging import PinnedStagingPool


def route_for(frames, tokens):
    batch, heads, width = frames.shape
    counts = (frames >= 0).sum(-1).unsqueeze(-1)
    return HistoryRoutePlan(method='test', routing_stage='pre-transfer',
        query_labels=torch.zeros(batch, heads, 2, dtype=torch.long),
        query_group_sizes=torch.full((batch, heads, 1), 2),
        union_frame_ids=frames, union_token_ids=tokens,
        group_union_indices=torch.arange(width).view(1, 1, 1, width).expand(batch, heads, 1, width),
        group_history_counts=counts, candidate_history_tokens=20, query_tokens=2,
        exact_k_tokens=0, target_history_density=.25)


@pytest.mark.parametrize('batch,heads', [(1, 1), (2, 3)])
def test_batched_requests_restore_same_noncanonical_order_tail_and_padding(batch, heads):
    torch.manual_seed(20260906)
    archive = HistoryArchive(SparseHistoryConfig(method='block64_history', block_size=4),
                             spatial_height=2, spatial_width=5)
    for f in (3, 8):
        k = torch.randn(batch, 10, heads, 8)
        archive.index_frame(2, f, k, k+9)
    frames = torch.tensor([8, 3, 8, 3, -1]).view(1, 1, 5).expand(batch, heads, 5).clone()
    tokens = torch.tensor([9, 2, 1, 7, -1]).view(1, 1, 5).expand_as(frames).clone()
    plan = route_for(frames, tokens)
    compiled = compile_raw_requests(plan, frame_tokens=10, block_tokens=4)
    assert compiled.blocks[:4] == ((0, 0, 8, 8, 10), (0, 0, 3, 0, 4),
                                  (0, 0, 8, 0, 4), (0, 0, 3, 4, 8))
    reference_cache, cache = RawHistoryBlockCache(100000), RawHistoryBlockCache(100000)
    pool = PinnedStagingPool(slots=2, budget_bytes=100000, pin_memory=False)
    before = plan.digest()
    for call in range(2):
        ref = archive.materialize_raw_block_cached(2, plan, reference_cache, device='cpu',
            current_frame_id=11+call, freqs=None, block_tokens=4, candidate_frame_ids=[3, 8])
        cur = archive.materialize_raw_block_cached(2, plan, cache, device='cpu',
            current_frame_id=11+call, freqs=None, block_tokens=4, candidate_frame_ids=[3, 8],
            implementation='batched', staging_pool=pool)
        assert torch.equal(ref.key_unrotated, cur.key_unrotated)
        assert torch.equal(ref.value, cur.value)
        assert torch.equal(ref.positions, cur.positions)
        assert ref.transferred_bytes == cur.transferred_bytes
        assert ref.cache_hit_bytes == cur.cache_hit_bytes
        assert cur.materialize_total_s >= cur.gpu_restore_s
        assert list(reference_cache._entries) == list(cache._entries)
        assert all(e.key_unrotated.untyped_storage().nbytes() == e.key_unrotated.numel()*e.key_unrotated.element_size()
                   for e in cache._entries.values())
        # New immutable archive frames do not invalidate old raw cache keys.
        k = torch.randn(batch, 10, heads, 8)
        archive.index_frame(2, 20+call, k, k+1)
    assert before == plan.digest()
    assert len(archive._raw_request_plans) == 1
    archive.clear_frames()
    assert not archive._raw_request_plans


def test_raw_compiler_rejects_invalid_coordinates_and_handles_empty_route():
    plan = route_for(torch.tensor([[[3]]]), torch.tensor([[[-1]]]))
    with pytest.raises(ValueError, match='within-frame'):
        compile_raw_requests(plan, frame_tokens=10, block_tokens=4)
    empty = replace(plan, union_frame_ids=torch.tensor([[[-1]]]))
    compiled = compile_raw_requests(empty, frame_tokens=10, block_tokens=4)
    assert compiled.blocks == ()
    assert compiled.source_indices.numel() == 0


def test_raw_batched_survives_lru_eviction_without_hiding_shared_slab():
    archive = HistoryArchive(SparseHistoryConfig(method='block64_history', block_size=2),
                             spatial_height=1, spatial_width=6)
    key = torch.arange(12).float().view(1, 6, 1, 2)
    archive.index_frame(0, 1, key, key+20)
    route = route_for(torch.ones(1, 1, 6, dtype=torch.long), torch.arange(6).view(1, 1, 6))
    cache = RawHistoryBlockCache(32)  # One block; current union may exceed persistent cache.
    result = archive.materialize_raw_block_cached(0, route, cache, device='cpu',
        current_frame_id=2, freqs=None, block_tokens=2, implementation='batched')
    assert torch.equal(result.key_unrotated, key)
    assert torch.equal(result.value, key+20)
    assert cache.current_bytes == 32
    assert len(cache._entries) == 1
    entry = next(iter(cache._entries.values()))
    assert entry.key_unrotated.untyped_storage().nbytes()+entry.value.untyped_storage().nbytes() == 32
