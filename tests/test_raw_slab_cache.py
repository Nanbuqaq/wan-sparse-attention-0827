import pytest
import torch

from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.raw_slab_cache import RawTokenSlabCache
from test_raw_composition import route_for


def test_partial_token_validity_never_loads_unselected_neighbors():
    archive = HistoryArchive(SparseHistoryConfig(method='block64_history', block_size=4),
                             spatial_height=1, spatial_width=10)
    k = torch.arange(40).float().view(1, 10, 2, 2)
    archive.index_frame(0, 5, k, k+100)
    cache = RawTokenSlabCache(512, head_dim=2, dtype=torch.float32, device='cpu', block_tokens=4)
    # Each phase extends or reuses different token subsets inside the same blocks.
    cases = [([0, 2, 9], [1, 8, 9], 6), ([2, 3, 9], [1, 7, 8], 2), ([0, 3, 9], [7, 8, 9], 0)]
    for left, right, miss_tokens in cases:
        tokens = torch.tensor([[left, right]])
        frames = torch.full_like(tokens, 5)
        route = route_for(frames, tokens)
        before = route.digest()
        result = archive.materialize_raw_block_cached(0, route, cache, device='cpu',
            current_frame_id=12, freqs=None, block_tokens=4, implementation='slab')
        expected = torch.stack([k[0, tokens[0, h], h] for h in range(2)], dim=1).unsqueeze(0)
        assert torch.equal(result.key_unrotated, expected)
        assert torch.equal(result.value, expected+100)
        assert result.transferred_bytes == miss_tokens*16
        assert result.padding_bytes == 0
        assert route.digest() == before
        assert cache.as_dict()['allocated_backing_bytes'] <= cache.budget_bytes


def test_slab_budget_protection_eviction_and_archive_epoch():
    archive = HistoryArchive(SparseHistoryConfig(method='block64_history', block_size=2),
                             spatial_height=1, spatial_width=4)
    cache = RawTokenSlabCache(32, head_dim=2, dtype=torch.float32, device='cpu', block_tokens=2)
    k = torch.arange(8).float().view(1, 4, 1, 2)
    archive.index_frame(0, 1, k, k+1)
    def run(tokens):
        route = route_for(torch.ones(1, 1, len(tokens), dtype=torch.long), torch.tensor([[tokens]]))
        return archive.materialize_raw_block_cached(0, route, cache, device='cpu',
            current_frame_id=5, freqs=None, block_tokens=2, implementation='slab')
    assert run([0, 1]).transferred_bytes == 32
    assert run([2, 3]).transferred_bytes == 32
    assert cache.evictions == 1
    with pytest.raises(MemoryError, match='working set'):
        run([0, 2])
    assert run([2, 3]).transferred_bytes == 0
    archive.clear_frames()
    archive.index_frame(0, 1, k+30, k+31)
    result = run([2, 3])
    assert result.transferred_bytes == 32
    assert torch.equal(result.key_unrotated, (k+30)[:, 2:])
    assert cache.as_dict()['allocated_backing_bytes'] == 32


def test_uncommitted_slab_reservation_never_becomes_a_hit():
    from adapters.longlive_sparse.history_cache import RawHistoryBlockCacheKey
    key = RawHistoryBlockCacheKey(0, 0, 0, 1, 1, 0, 2, 'torch.float32', 'cpu')
    cache = RawTokenSlabCache(32, head_dim=2, dtype=torch.float32, device='cpu', block_tokens=2)
    assert cache.reserve([key], [1])[1] == [1]
    assert cache.reserve([key], [1])[1] == [1]
    cache.commit([key], [1])
    assert cache.reserve([key], [3])[1] == [2]
