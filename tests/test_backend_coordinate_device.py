from __future__ import annotations

import torch

from adapters.longlive_sparse.ar_routing import route_history
from adapters.longlive_sparse.route_plan import map_union_coordinates


def test_union_coordinate_mapping_returns_candidate_device_indices():
    generator = torch.Generator().manual_seed(20260828)
    query = torch.randn(1, 16, 1, 128, generator=generator)
    key = torch.randn(1, 24, 1, 128, generator=generator)
    frame_ids = torch.arange(3).repeat_interleave(8).view(1, 1, 24)
    token_ids = torch.arange(8).repeat(3).view(1, 1, 24)
    permutation = torch.tensor(
        [16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    )
    key = key.index_select(1, permutation)
    frame_ids = frame_ids.index_select(2, permutation)
    token_ids = token_ids.index_select(2, permutation)
    plan = route_history(
        query,
        key,
        frame_ids,
        token_ids,
        method="block64_history",
        density=0.25,
        exact_k_tokens=8,
    )

    indices = map_union_coordinates(plan, frame_ids, token_ids)

    assert indices.device == frame_ids.device
    valid = plan.union_frame_ids >= 0
    assert torch.equal(
        frame_ids.gather(-1, indices)[valid], plan.union_frame_ids[valid]
    )
    assert torch.equal(
        token_ids.gather(-1, indices)[valid], plan.union_token_ids[valid]
    )
