import pytest
import torch

from adapters.longlive_sparse.history_cache import HierarchicalHistoryCache
from adapters.longlive_sparse.system_config import LongLiveSystemConfig


def test_hierarchical_budget_and_chunk_clear_keep_raw_backing_counted():
    cache = HierarchicalHistoryCache(1024, 512)
    raw = cache.raw(head_dim=2, dtype=torch.float32, device='cpu', block_tokens=4)
    raw.allocate()
    cache.begin_chunk(1, per_chunk=True)
    cache.begin_chunk(2, per_chunk=True)
    state = cache.as_dict()
    assert state['budget_bytes'] == 1024
    assert state['union_budget_bytes'] == 512
    assert state['raw_slab']['allocated_backing_bytes'] == 512
    assert state['current_bytes'] == 512
    assert cache.raw_cache is raw


def test_hierarchical_config_freezes_both_budgets_and_rejects_ambiguous_layout():
    config = LongLiveSystemConfig(gpu_union_cache='hierarchical', gpu_union_cache_budget_mib=4096,
        raw_cache_budget_mib=1024, transfer_layout='exact_compact')
    assert config.identity_dict()['raw_cache_budget_mib'] == 1024
    with pytest.raises(ValueError, match='smaller'):
        LongLiveSystemConfig(gpu_union_cache='hierarchical', gpu_union_cache_budget_mib=1024,
            raw_cache_budget_mib=1024, transfer_layout='exact_compact')
    with pytest.raises(ValueError, match='only valid'):
        LongLiveSystemConfig(raw_cache_budget_mib=1024)
