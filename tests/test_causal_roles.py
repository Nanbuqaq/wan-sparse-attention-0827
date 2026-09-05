from __future__ import annotations

import pytest
import torch

from adapters.longlive_sparse.causal_roles import (
    CausalSubjectRouter,
    align_pixel_patch_masks_to_latent,
    build_block_role_log_prior,
    build_identity_scene_bias_plan,
    patch_masks_to_block_identity,
    soft_role_agreement,
)
from adapters.longlive_sparse.contexts import OnlineRoutingContext
from adapters.longlive_sparse.route_plan import HistoryRoutePlan


def test_pixel_masks_use_explicit_stride_four_latent_anchors() -> None:
    masks = torch.arange(13).view(13, 1, 1).expand(13, 30, 52)
    latent = align_pixel_patch_masks_to_latent(masks, latent_frames=4)
    assert latent[:, 0, 0].tolist() == [0.0, 4.0, 8.0, 12.0]


def test_patch_mask_block_probabilities_keep_partial_last_block() -> None:
    masks = torch.zeros((1, 30, 52))
    masks[:, :, :26] = 1
    probability, starts, ends = patch_masks_to_block_identity(masks)
    assert probability.shape == (1, 25)
    assert starts[0] == 0 and ends[-1] == 1560


def _context(frame_ids: torch.Tensor) -> OnlineRoutingContext:
    blocks = frame_ids.numel()
    starts = torch.arange(blocks) % 25 * 64
    ends = (starts + 64).clamp_max(1560)
    return OnlineRoutingContext(
        query_centroids=torch.randn(1, 2, 3, 4),
        query_group_sizes=torch.ones(1, 2, 3, dtype=torch.long),
        key_prototypes=torch.randn(1, 2, blocks, 4),
        value_prototypes=torch.randn(1, 2, blocks, 4),
        block_frame_ids=frame_ids,
        block_token_starts=starts,
        block_token_ends=ends,
        block_age=torch.ones(blocks),
    )


def test_causal_router_updates_only_after_contiguous_completed_chunks() -> None:
    router = CausalSubjectRouter()
    first = torch.zeros((2, 30, 52))
    first[:, :, :26] = 1
    router.commit_completed_chunk(
        first,
        start_latent_frame=0,
        refresh_service_s=1.0,
        vae_decode_service_s=0.5,
    )
    result = router.build_roles(_context(torch.tensor([0, 1])))
    assert result.query_role_probabilities.shape == (1, 3, 2)
    assert result.history_role_probabilities.shape == (1, 2, 2, 2)
    torch.testing.assert_close(
        result.query_role_probabilities.sum(dim=-1), torch.ones(1, 3)
    )
    assert result.metadata["current_or_future_mask_read"] is False
    with pytest.raises(ValueError, match="current or future"):
        router.build_roles(_context(torch.tensor([0, 2])))
    with pytest.raises(ValueError, match="contiguously"):
        router.commit_completed_chunk(first[:1], start_latent_frame=3)


def test_soft_role_agreement_reports_error_and_threshold_match() -> None:
    metrics = soft_role_agreement(
        torch.tensor([0.1, 0.9]), torch.tensor([0.2, 0.8])
    )
    assert metrics["mean_absolute_error"] == pytest.approx(0.1)
    assert metrics["binary_agreement"] == 1.0


def test_causal_roles_map_to_compact_attention_bias_plan() -> None:
    router = CausalSubjectRouter()
    masks = torch.zeros((2, 30, 52))
    masks[:, :, :26] = 1
    router.commit_completed_chunk(masks, start_latent_frame=0)
    context = _context(torch.tensor([0, 1]))
    roles = router.build_roles(context)
    route = HistoryRoutePlan(
        method="test",
        routing_stage="pre-transfer",
        query_labels=torch.tensor([[[0, 1, 2], [0, 1, 2]]]),
        query_group_sizes=torch.ones((1, 2, 3), dtype=torch.long),
        union_frame_ids=torch.tensor([[[0, 1], [0, 1]]]),
        union_token_ids=torch.tensor([[[0, 64], [0, 64]]]),
        group_union_indices=torch.tensor(
            [[
                [[0, 1], [0, 1], [0, 1]],
                [[0, 1], [0, 1], [0, 1]],
            ]]
        ),
        group_history_counts=torch.full((1, 2, 3), 2, dtype=torch.long),
        candidate_history_tokens=3120,
        query_tokens=3,
        exact_k_tokens=8,
        target_history_density=0.25,
    )
    bias = build_identity_scene_bias_plan(
        route, context, roles, context_weight=0.2
    )
    assert bias.query_role_probabilities.shape == (1, 3, 2)
    assert bias.history_role_probabilities.shape == (1, 2, 2, 2)
    assert bias.metadata["dense_qk_bias_materialized"] is False
    prior = build_block_role_log_prior(context, roles, context_weight=0.2)
    assert prior.shape == (1, 2, 3, 2)
    assert torch.isfinite(prior).all()
