import pytest
import torch
from adapters.longlive_sparse.stable_source_mask_fill import stable_mask_indices


def test_budget_coverage_and_small_foreground_changes_do_not_reshuffle_fill():
    generator = torch.Generator().manual_seed(91)
    for size in (29, 64, 880, 7040):
        budget = size // 2
        ids = torch.randperm(size, generator=generator)[:budget // 2]
        route = stable_mask_indices(ids, source_tokens=size, budget=budget)
        assert route.numel() == route.unique().numel() == budget
        assert set(ids.tolist()) <= set(route.tolist())
        assert torch.equal(route, stable_mask_indices(ids.flip(0), source_tokens=size, budget=budget))
        smaller = stable_mask_indices(ids[:-1], source_tokens=size, budget=budget)
        assert len(set(route.tolist()) - set(smaller.tolist())) <= 1
        assert len(set(smaller.tolist()) - set(route.tolist())) <= 1


def test_invalid_masks_fail_without_trimming():
    for ids, budget in (([1, 1], 2), ([], 2), ([-1], 2), ([8], 2), ([1, 2, 3], 2)):
        with pytest.raises(ValueError):
            stable_mask_indices(ids, source_tokens=8, budget=budget)
