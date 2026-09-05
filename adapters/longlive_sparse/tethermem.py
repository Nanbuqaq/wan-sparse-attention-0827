"""Causal-safe TetherMem primitives adapted from the pinned MIT release."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class TetherMethodContract:
    method_id: str
    causal_online: bool
    uses_full_reference_video: bool
    online_speed_pareto_eligible: bool
    mask_update_boundary: str

    def __post_init__(self) -> None:
        if self.uses_full_reference_video and self.causal_online:
            raise ValueError("a full-reference Tether method cannot be causal online")
        if self.online_speed_pareto_eligible and not self.causal_online:
            raise ValueError("only causal online Tether methods can enter speed Pareto")

    def as_dict(self) -> dict:
        return asdict(self)


TETHER_METHODS = {
    "tethermem_oracle_mask_teacher": TetherMethodContract(
        method_id="tethermem_oracle_mask_teacher",
        causal_online=False,
        uses_full_reference_video=True,
        online_speed_pareto_eligible=False,
        mask_update_boundary="full matched RAG-Dense video before routed generation",
    ),
    "causal_subject_router": TetherMethodContract(
        method_id="causal_subject_router",
        causal_online=True,
        uses_full_reference_video=False,
        online_speed_pareto_eligible=True,
        mask_update_boundary="completed chunk only",
    ),
    "causal_subject_final_history": TetherMethodContract(
        method_id="causal_subject_final_history",
        causal_online=True,
        uses_full_reference_video=False,
        online_speed_pareto_eligible=True,
        mask_update_boundary="completed chunk only; shared union remains at most 25 percent",
    ),
}


def tether_method_contract(method_id: str) -> TetherMethodContract:
    try:
        return TETHER_METHODS[method_id]
    except KeyError as error:
        raise ValueError(f"unknown Tether method: {method_id!r}") from error


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
