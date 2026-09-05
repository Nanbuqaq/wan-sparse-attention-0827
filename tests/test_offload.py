import torch
import pytest
from adapters.longlive_sparse.offload import ArchiveOffloadStager
from adapters.longlive_sparse.staging import PinnedStagingPool


def test_pageable_commit_is_independent_of_reused_ring_and_ticket_cannot_repeat():
    pool=PinnedStagingPool(slots=2,budget_bytes=8192,pin_memory=False)
    stager=ArchiveOffloadStager(pool)
    key=torch.arange(128).reshape(2,2,4,8).float()
    value=-key
    ticket=stager.launch(key,value)
    archived_k,archived_v,metrics=stager.complete(ticket)
    assert torch.equal(archived_k,key) and torch.equal(archived_v,value)
    assert not metrics['archive_is_pinned']
    assert metrics['payload_bytes']==1024
    with pytest.raises(ValueError):
        stager.complete(ticket)
    other=stager.launch(key+10,value+20)
    stager.complete(other)
    assert torch.equal(archived_k,key) and torch.equal(archived_v,value)


def test_pool_keeps_two_alternating_shapes_without_pin_allocation_thrash():
    pool=PinnedStagingPool(slots=2,budget_bytes=4096,pin_memory=False)
    for _ in range(5):
        for shape in ((1,16),(1,32)):
            lease=pool.acquire(shape,torch.float32,fused=False)
            pool.release(lease)
    assert pool.allocations==2 and pool.reuses==8


def test_pool_reclaims_free_cached_shape_if_empty_slot_would_exceed_budget():
    pool=PinnedStagingPool(slots=2,budget_bytes=256,pin_memory=False)
    lease=pool.acquire((1,16),torch.float32,fused=False)
    pool.release(lease)
    lease=pool.acquire((1,32),torch.float32,fused=False)
    assert pool.as_dict()['allocated_bytes']==256
    pool.release(lease)


def test_growing_onload_shape_does_not_evict_recent_offload_shape():
    pool=PinnedStagingPool(slots=2,budget_bytes=4096,pin_memory=False)
    for shape in ((1,16),(1,4),(1,16),(1,8),(1,16),(1,8)):
        lease=pool.acquire(shape,torch.float32,fused=False)
        pool.release(lease)
    assert pool.allocations==3
