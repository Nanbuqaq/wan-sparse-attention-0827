"""Causal prototype novelty and redundancy for the CPU history bank."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_prototype_novelty(
    key_prototypes: torch.Tensor, block_frame_ids: torch.Tensor
) -> torch.Tensor:
    """Return `1 - max cosine to an earlier-frame block` for B/H/K prototypes."""

    if key_prototypes.ndim != 4:
        raise ValueError("key_prototypes must be [B,H,K,D]")
    if block_frame_ids.ndim != 1 or block_frame_ids.shape[0] != key_prototypes.shape[2]:
        raise ValueError("block_frame_ids must have one entry per prototype")
    normalized = F.normalize(key_prototypes.float(), dim=-1)
    novelty = torch.ones(key_prototypes.shape[:3], dtype=torch.float32)
    for block in range(key_prototypes.shape[2]):
        prior = torch.nonzero(
            block_frame_ids < block_frame_ids[block], as_tuple=False
        ).flatten()
        if not prior.numel():
            continue
        similarity = torch.einsum(
            "bhd,bhpd->bhp",
            normalized[:, :, block],
            normalized.index_select(2, prior),
        )
        novelty[:, :, block] = 1.0 - similarity.amax(dim=-1).clamp(-1.0, 1.0)
    return novelty.clamp(0.0, 2.0)


def combine_value_and_novelty(
    value: torch.Tensor, novelty: torch.Tensor, *, novelty_weight: float
) -> torch.Tensor:
    if value.shape != novelty.shape:
        raise ValueError("value and novelty must share B/H/K shape")
    if novelty_weight < 0:
        raise ValueError("novelty_weight must be non-negative")
    value_scale = value.float() / value.float().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    novelty_scale = novelty.float() / novelty.float().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    return value_scale + novelty_weight * novelty_scale
