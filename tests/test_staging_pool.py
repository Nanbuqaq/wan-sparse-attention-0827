from __future__ import annotations

from pathlib import Path

import pytest
import torch

from adapters.longlive_sparse.staging import PinnedStagingPool


def test_staging_pool_reuses_matching_separate_buffers() -> None:
    pool = PinnedStagingPool(slots=1, budget_bytes=4096, pin_memory=False)
    first = pool.acquire((1, 8, 2, 4), torch.float32, fused=False)
    first.key.fill_(3)
    pointer = first.key.data_ptr()
    pool.release(first)
    second = pool.acquire((1, 8, 2, 4), torch.float32, fused=False)
    assert second.reused is True
    assert second.key.data_ptr() == pointer
    pool.release(second)
    assert pool.as_dict()["reuses"] == 1


def test_staging_pool_fused_views_share_one_allocation() -> None:
    pool = PinnedStagingPool(slots=1, budget_bytes=4096, pin_memory=False)
    lease = pool.acquire((1, 8, 2, 4), torch.float32, fused=True)
    assert lease.fused is not None
    lease.key.fill_(1)
    lease.value.fill_(2)
    assert torch.equal(lease.fused[0], lease.key)
    assert torch.equal(lease.fused[1], lease.value)
    pool.release(lease)


def test_staging_pool_fails_closed_on_budget_or_double_release() -> None:
    pool = PinnedStagingPool(slots=1, budget_bytes=8, pin_memory=False)
    with pytest.raises(MemoryError, match="exceeds"):
        pool.acquire((1, 8, 2, 4), torch.float32, fused=False)
    pool = PinnedStagingPool(slots=1, budget_bytes=4096, pin_memory=False)
    lease = pool.acquire((1, 1, 1, 1), torch.float32, fused=False)
    pool.release(lease)
    with pytest.raises(ValueError, match="not active"):
        pool.release(lease)


def test_archive_persistent_staging_uses_materialized_tensor_dtype() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "adapters/longlive_sparse/archive.py"
    ).read_text(encoding="utf-8")
    assert "tuple(physical_key.shape), physical_key.dtype" in source
