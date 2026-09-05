"""Causal-safe TetherMem primitives adapted from the pinned MIT release."""

from __future__ import annotations

import torch


def solve_context_weight(
    subject_fraction: float, anchor_weight: float = 1.0, target_average: float = 0.25
) -> float:
    if not 0.0 <= subject_fraction <= 1.0:
        raise ValueError("subject_fraction must be in [0,1]")
    if anchor_weight < 0.0 or target_average < 0.0:
        raise ValueError("weights must be non-negative")
    if 1.0 - subject_fraction > 1e-6:
        value = (
            target_average - subject_fraction * anchor_weight
        ) / (1.0 - subject_fraction)
    else:
        value = target_average
    return min(max(value, 0.0), 1.0)


def soft_region_age_prior(
    query_role_probabilities: torch.Tensor,
    history_role_probabilities: torch.Tensor,
    history_age_weights: torch.Tensor,
    *,
    context_weight: float,
) -> torch.Tensor:
    """Return expected identity/scene prior as compact B/H/G/K weights.

    Role order is `(identity, scene)`.  Same identity access has weight one,
    cross-region access uses `context_weight`, and scene-to-scene access uses
    the supplied causal age weight.
    """

    if query_role_probabilities.ndim != 3 or query_role_probabilities.shape[-1] != 2:
        raise ValueError("query roles must be [B,G,2]")
    if history_role_probabilities.ndim != 4 or history_role_probabilities.shape[-1] != 2:
        raise ValueError("history roles must be [B,H,K,2]")
    if history_age_weights.shape != history_role_probabilities.shape[:-1]:
        raise ValueError("age weights must match history B/H/K axes")
    if not 0.0 <= context_weight <= 1.0:
        raise ValueError("context_weight must be in [0,1]")
    if bool((history_age_weights <= 0).any()):
        raise ValueError("history age weights must be positive")
    query = query_role_probabilities[:, None, :, None, :]
    history = history_role_probabilities[:, :, None, :, :]
    q_identity, q_scene = query[..., 0], query[..., 1]
    k_identity, k_scene = history[..., 0], history[..., 1]
    prior = (
        q_identity * k_identity
        + context_weight * q_identity * k_scene
        + context_weight * q_scene * k_identity
        + q_scene * k_scene * history_age_weights[:, :, None, :]
    )
    return prior.clamp_min(1e-9)
