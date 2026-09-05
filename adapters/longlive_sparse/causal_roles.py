"""Causal subject-role state updated only after completed latent chunks."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .attention_bias import AttentionBiasPlan
from .contexts import OnlineRoutingContext
from .memory_roles import causal_query_identity_probability
from .route_plan import HistoryRoutePlan
from .tethermem import soft_region_age_prior


PATCH_HEIGHT = 30
PATCH_WIDTH = 52


def align_pixel_patch_masks_to_latent(
    pixel_patch_masks: torch.Tensor,
    *,
    latent_frames: int,
) -> torch.Tensor:
    """Map `4*T-3` decoded patch masks to explicit latent anchors 0,4,... ."""

    if pixel_patch_masks.ndim != 3 or tuple(pixel_patch_masks.shape[1:]) != (
        PATCH_HEIGHT,
        PATCH_WIDTH,
    ):
        raise ValueError("pixel patch masks must be [4*T-3,30,52]")
    if latent_frames < 1:
        raise ValueError("latent_frames must be positive")
    expected = 4 * latent_frames - 3
    if pixel_patch_masks.shape[0] != expected:
        raise ValueError(
            f"decoded mask count {pixel_patch_masks.shape[0]} != 4*T-3 ({expected})"
        )
    indices = torch.arange(latent_frames, dtype=torch.long) * 4
    return pixel_patch_masks.float().index_select(0, indices)


def patch_masks_to_block_identity(
    latent_patch_masks: torch.Tensor,
    *,
    block_tokens: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return soft identity probability and within-frame block coordinates."""

    if latent_patch_masks.ndim != 3 or tuple(latent_patch_masks.shape[1:]) != (
        PATCH_HEIGHT,
        PATCH_WIDTH,
    ):
        raise ValueError("latent patch masks must be [T,30,52]")
    if block_tokens < 1:
        raise ValueError("block_tokens must be positive")
    flattened = latent_patch_masks.float().clamp(0.0, 1.0).flatten(1)
    starts = torch.arange(0, PATCH_HEIGHT * PATCH_WIDTH, block_tokens)
    ends = (starts + block_tokens).clamp_max(PATCH_HEIGHT * PATCH_WIDTH)
    probabilities = torch.stack(
        [flattened[:, start:end].mean(dim=-1) for start, end in zip(starts, ends)],
        dim=-1,
    )
    return probabilities, starts, ends


@dataclass(frozen=True)
class CausalRoleResult:
    query_role_probabilities: torch.Tensor
    history_role_probabilities: torch.Tensor
    committed_latent_frames: int
    metadata: dict


class CausalSubjectRouter:
    """State machine whose role evidence contains completed frames only."""

    def __init__(self, *, block_tokens: int = 64):
        if block_tokens < 1:
            raise ValueError("block_tokens must be positive")
        self.block_tokens = int(block_tokens)
        self._latent_masks = torch.empty(
            (0, PATCH_HEIGHT, PATCH_WIDTH), dtype=torch.float32
        )
        self.refresh_service_s = 0.0
        self.vae_decode_service_s = 0.0
        self.synchronization_service_s = 0.0

    @property
    def committed_latent_frames(self) -> int:
        return int(self._latent_masks.shape[0])

    def commit_completed_chunk(
        self,
        masks: torch.Tensor,
        *,
        start_latent_frame: int,
        refresh_service_s: float = 0.0,
        vae_decode_service_s: float = 0.0,
        synchronization_service_s: float = 0.0,
    ) -> None:
        if start_latent_frame != self.committed_latent_frames:
            raise ValueError("completed mask chunks must commit contiguously")
        if masks.ndim != 3 or tuple(masks.shape[1:]) != (
            PATCH_HEIGHT,
            PATCH_WIDTH,
        ):
            raise ValueError("committed masks must be latent [T,30,52]")
        for name, value in (
            ("refresh_service_s", refresh_service_s),
            ("vae_decode_service_s", vae_decode_service_s),
            ("synchronization_service_s", synchronization_service_s),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        self._latent_masks = torch.cat(
            (self._latent_masks, masks.detach().to("cpu").float().clamp(0.0, 1.0)),
            dim=0,
        )
        self.refresh_service_s += float(refresh_service_s)
        self.vae_decode_service_s += float(vae_decode_service_s)
        self.synchronization_service_s += float(synchronization_service_s)

    def build_roles(
        self,
        context: OnlineRoutingContext,
        *,
        previous_query_spatial_prior: torch.Tensor | None = None,
    ) -> CausalRoleResult:
        if self.committed_latent_frames == 0:
            raise RuntimeError("causal subject router has no completed chunk")
        if bool((context.block_frame_ids >= self.committed_latent_frames).any()):
            raise ValueError("online role routing cannot read current or future frame masks")
        block_probability, starts, ends = patch_masks_to_block_identity(
            self._latent_masks, block_tokens=self.block_tokens
        )
        start_to_block = {int(start): index for index, start in enumerate(starts)}
        identity = torch.empty(
            context.key_prototypes.shape[:3], dtype=torch.float32
        )
        for block in range(context.blocks):
            frame_id = int(context.block_frame_ids[block])
            token_start = int(context.block_token_starts[block])
            try:
                within_frame_block = start_to_block[token_start]
            except KeyError as error:
                raise ValueError(
                    "online block starts do not match the causal mask Block64 grid"
                ) from error
            identity[:, :, block] = block_probability[
                frame_id, within_frame_block
            ]
        query_identity_by_head = causal_query_identity_probability(
            context.query_centroids,
            context.key_prototypes,
            identity,
            previous_spatial_prior=previous_query_spatial_prior,
        )
        query_identity = query_identity_by_head.mean(dim=1)
        query_roles = torch.stack((query_identity, 1.0 - query_identity), dim=-1)
        history_roles = torch.stack((identity, 1.0 - identity), dim=-1)
        return CausalRoleResult(
            query_role_probabilities=query_roles,
            history_role_probabilities=history_roles,
            committed_latent_frames=self.committed_latent_frames,
            metadata={
                "router": "causal_subject_router",
                "mask_timeline": "latent anchors sampled from decoded frames 0,4,...",
                "current_or_future_mask_read": False,
                "block_tokens": self.block_tokens,
                "mask_block_starts": starts.tolist(),
                "mask_block_ends": ends.tolist(),
                "refresh_service_s": self.refresh_service_s,
                "vae_decode_service_s": self.vae_decode_service_s,
                "synchronization_service_s": self.synchronization_service_s,
            },
        )

    def as_dict(self) -> dict:
        return {
            "router": "causal_subject_router",
            "committed_latent_frames": self.committed_latent_frames,
            "block_tokens": self.block_tokens,
            "refresh_service_s": self.refresh_service_s,
            "vae_decode_service_s": self.vae_decode_service_s,
            "synchronization_service_s": self.synchronization_service_s,
        }


def soft_role_agreement(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    if left.shape != right.shape:
        raise ValueError("role probability tensors must share shape")
    left = left.float().clamp(0.0, 1.0)
    right = right.float().clamp(0.0, 1.0)
    return {
        "mean_absolute_error": float((left - right).abs().mean()),
        "binary_agreement": float(((left >= 0.5) == (right >= 0.5)).float().mean()),
    }


def build_identity_scene_bias_plan(
    route_plan: HistoryRoutePlan,
    context: OnlineRoutingContext,
    roles: CausalRoleResult,
    *,
    context_weight: float,
    age_decay_floor: float = 0.05,
) -> AttentionBiasPlan:
    """Map compact group/block roles onto route Q/union axes without `[Q,K]`."""

    if not 0.0 <= context_weight <= 1.0:
        raise ValueError("context_weight must be in [0,1]")
    if not 0.0 < age_decay_floor <= 1.0:
        raise ValueError("age_decay_floor must be in (0,1]")
    group_sizes = context.query_group_sizes.long()
    if bool((group_sizes.sum(dim=-1) != route_plan.query_tokens).any()):
        raise ValueError("query groups must cover the route query length")
    if bool((group_sizes != group_sizes[:, :1]).any()):
        raise ValueError("head-averaged query roles require shared query group sizes")
    query_parts = []
    for batch in range(group_sizes.shape[0]):
        query_parts.append(
            torch.repeat_interleave(
                roles.query_role_probabilities[batch],
                group_sizes[batch, 0],
                dim=0,
            )
        )
    query_roles = torch.stack(query_parts, dim=0)
    history_roles = torch.zeros(
        (*route_plan.union_frame_ids.shape, 2), dtype=torch.float32
    )
    age = torch.ones(route_plan.union_frame_ids.shape, dtype=torch.float32)
    max_age = context.block_age.float().max().clamp_min(1.0)
    for batch in range(route_plan.union_frame_ids.shape[0]):
        for head in range(route_plan.union_frame_ids.shape[1]):
            for union in range(route_plan.union_frame_ids.shape[2]):
                frame = int(route_plan.union_frame_ids[batch, head, union])
                token = int(route_plan.union_token_ids[batch, head, union])
                if frame < 0:
                    history_roles[batch, head, union, 1] = 1.0
                    continue
                matches = torch.nonzero(
                    (context.block_frame_ids == frame)
                    & (context.block_token_starts <= token)
                    & (context.block_token_ends > token),
                    as_tuple=False,
                ).flatten()
                if matches.numel() != 1:
                    raise KeyError("each route coordinate must map to one role block")
                block = int(matches[0])
                history_roles[batch, head, union] = roles.history_role_probabilities[
                    batch, head, block
                ]
                age[batch, head, union] = max(
                    1.0 - float(context.block_age[block]) / float(max_age),
                    age_decay_floor,
                )
    return AttentionBiasPlan(
        role_names=("identity", "scene"),
        query_role_probabilities=query_roles,
        history_role_probabilities=history_roles,
        history_age_weights=age,
        mode="causal_subject_router",
        metadata={
            **roles.metadata,
            "context_weight": context_weight,
            "age_decay_floor": age_decay_floor,
            "dense_qk_bias_materialized": False,
            "route_plan_sha256": route_plan.digest(),
        },
    )


def build_block_role_log_prior(
    context: OnlineRoutingContext,
    roles: CausalRoleResult,
    *,
    context_weight: float,
    age_decay_floor: float = 0.05,
) -> torch.Tensor:
    """Return compact causal B/H/G/Block log prior for utility admission."""

    if not 0.0 <= context_weight <= 1.0:
        raise ValueError("context_weight must be in [0,1]")
    if not 0.0 < age_decay_floor <= 1.0:
        raise ValueError("age_decay_floor must be in (0,1]")
    max_age = context.block_age.float().max().clamp_min(1.0)
    age = (
        1.0 - context.block_age.float() / max_age
    ).clamp_min(age_decay_floor)
    age = age.view(1, 1, -1).expand(context.key_prototypes.shape[:3])
    prior = soft_region_age_prior(
        roles.query_role_probabilities,
        roles.history_role_probabilities,
        age,
        context_weight=context_weight,
    )
    return prior.log()
