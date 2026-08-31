from __future__ import annotations

import torch

from adapters.longlive_sparse.ar_routing import route_history
from scripts.calibrate_methods_from_trace import _selected_candidate_indices
from scripts.calibrate_proposed_history_from_trace import (
    candidate_coordinate_index,
    selected_candidate_indices,
)


def test_calibration_coordinate_lookup_handles_unsorted_candidates():
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
        method="svg2_ar",
        density=0.25,
        exact_k_tokens=8,
        spec_override={"q_clusters": 4, "k_clusters": 6, "iterations": 2},
    )

    group = int(plan.query_labels[0, 0, 0])
    indices = _selected_candidate_indices(
        plan, frame_ids, token_ids, 0, 0, group
    )
    count = int(plan.group_history_counts[0, 0, group])
    union_slots = plan.group_union_indices[0, 0, group, :count]

    assert torch.equal(
        frame_ids[0, 0].index_select(0, indices),
        plan.union_frame_ids[0, 0].index_select(0, union_slots),
    )
    assert torch.equal(
        token_ids[0, 0].index_select(0, indices),
        plan.union_token_ids[0, 0].index_select(0, union_slots),
    )


def test_proposed_teacher_batched_lookup_handles_unsorted_candidates():
    generator = torch.Generator().manual_seed(20260831)
    query = torch.randn(1, 16, 1, 128, generator=generator)
    key = torch.randn(1, 24, 1, 128, generator=generator)
    frame_ids = torch.arange(3).repeat_interleave(8).view(1, 1, 24)
    token_ids = torch.arange(8).repeat(3).view(1, 1, 24)
    permutation = torch.tensor(
        [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
         0, 1, 2, 3, 4, 5, 6, 7]
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
    teacher = {
        **candidate_coordinate_index(
            {"frame_ids": frame_ids, "token_ids": token_ids}
        ),
        "sample_ids": torch.tensor([0, 7, 15]),
    }
    selected = selected_candidate_indices(teacher, plan)
    for sample_offset, query_id in enumerate(teacher["sample_ids"].tolist()):
        group = int(plan.query_labels[0, 0, query_id])
        count = int(plan.group_history_counts[0, 0, group])
        union_slots = plan.group_union_indices[0, 0, group, :count]
        assert torch.equal(
            frame_ids[0, 0].index_select(0, selected[0, 0, sample_offset]),
            plan.union_frame_ids[0, 0].index_select(0, union_slots),
        )
        assert torch.equal(
            token_ids[0, 0].index_select(0, selected[0, 0, sample_offset]),
            plan.union_token_ids[0, 0].index_select(0, union_slots),
        )
