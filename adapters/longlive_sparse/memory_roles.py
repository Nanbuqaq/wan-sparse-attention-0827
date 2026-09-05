"""Causal identity/scene/state role probabilities for historical Block64s."""

from __future__ import annotations

import torch


def causal_state_change_score(
    value_prototypes: torch.Tensor,
    block_frame_ids: torch.Tensor,
    block_token_starts: torch.Tensor,
) -> torch.Tensor:
    """Compare each block only with the latest matching spatial block in the past."""

    if value_prototypes.ndim != 4:
        raise ValueError("value_prototypes must be [B,H,K,D]")
    blocks = value_prototypes.shape[2]
    if block_frame_ids.shape != (blocks,) or block_token_starts.shape != (blocks,):
        raise ValueError("frame ids and token starts must have one value per block")
    score = torch.zeros(value_prototypes.shape[:3], dtype=torch.float32)
    values = value_prototypes.float()
    for block in range(blocks):
        prior = torch.nonzero(
            (block_frame_ids < block_frame_ids[block])
            & (block_token_starts == block_token_starts[block]),
            as_tuple=False,
        ).flatten()
        if not prior.numel():
            continue
        latest_frame = block_frame_ids.index_select(0, prior).max()
        latest = prior[
            block_frame_ids.index_select(0, prior) == latest_frame
        ][-1]
        delta = values[:, :, block] - values[:, :, latest]
        score[:, :, block] = torch.linalg.vector_norm(delta, dim=-1)
    scale = score.amax(dim=-1, keepdim=True).clamp_min(1e-12)
    return score / scale


def build_three_role_probabilities(
    identity_probability: torch.Tensor,
    state_probability: torch.Tensor,
) -> torch.Tensor:
    """Return normalized `(identity, scene, state)` probabilities."""

    if identity_probability.shape != state_probability.shape:
        raise ValueError("identity and state probabilities must share shape")
    if identity_probability.ndim != 3:
        raise ValueError("role inputs must be [B,H,K]")
    identity = identity_probability.float().clamp(0.0, 1.0)
    state = state_probability.float().clamp(0.0, 1.0) * (1.0 - identity)
    scene = (1.0 - identity - state).clamp_min(0.0)
    result = torch.stack((identity, scene, state), dim=-1)
    return result / result.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def causal_query_identity_probability(
    query_centroids: torch.Tensor,
    key_prototypes: torch.Tensor,
    block_identity_probability: torch.Tensor,
    *,
    previous_spatial_prior: torch.Tensor | None = None,
) -> torch.Tensor:
    """Predict soft query identity roles from current Q and past prototypes."""

    if query_centroids.ndim != 4 or key_prototypes.ndim != 4:
        raise ValueError("query/key prototypes must be [B,H,G/K,D]")
    if query_centroids.shape[:2] != key_prototypes.shape[:2]:
        raise ValueError("query/key prototypes must share B/H")
    if block_identity_probability.shape != key_prototypes.shape[:3]:
        raise ValueError("block identity probabilities must be [B,H,K]")
    identity_weight = block_identity_probability.float()
    scene_weight = 1.0 - identity_weight
    identity_centroid = (
        key_prototypes.float() * identity_weight.unsqueeze(-1)
    ).sum(dim=2) / identity_weight.sum(dim=2, keepdim=True).clamp_min(1e-6)
    scene_centroid = (
        key_prototypes.float() * scene_weight.unsqueeze(-1)
    ).sum(dim=2) / scene_weight.sum(dim=2, keepdim=True).clamp_min(1e-6)
    query = torch.nn.functional.normalize(query_centroids.float(), dim=-1)
    identity_centroid = torch.nn.functional.normalize(identity_centroid, dim=-1)
    scene_centroid = torch.nn.functional.normalize(scene_centroid, dim=-1)
    identity_score = torch.einsum("bhgd,bhd->bhg", query, identity_centroid)
    scene_score = torch.einsum("bhgd,bhd->bhg", query, scene_centroid)
    probability = torch.sigmoid(identity_score - scene_score)
    if previous_spatial_prior is not None:
        try:
            prior = torch.broadcast_to(previous_spatial_prior.float(), probability.shape)
        except RuntimeError as error:
            raise ValueError("previous_spatial_prior is not broadcastable to B/H/G") from error
        probability = 0.5 * probability + 0.5 * prior.clamp(0.0, 1.0)
    return probability
