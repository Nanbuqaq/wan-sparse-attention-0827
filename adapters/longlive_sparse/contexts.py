"""Explicit online and offline routing information boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


def _require_tensor(name: str, value: torch.Tensor, ndim: int) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D tensor")


@dataclass(frozen=True)
class OnlineRoutingContext:
    """Causal inputs available before raw historical KV materialization.

    The type intentionally has no dense output, full candidate K/V, or current
    attention-weight fields.  Past statistics are immutable snapshots produced
    by completed calls only.
    """

    query_centroids: torch.Tensor
    query_group_sizes: torch.Tensor
    key_prototypes: torch.Tensor
    value_prototypes: torch.Tensor
    block_frame_ids: torch.Tensor
    block_token_starts: torch.Tensor
    block_token_ends: torch.Tensor
    block_age: torch.Tensor
    past_attention_score: torch.Tensor | None = None
    query_role_probabilities: torch.Tensor | None = None
    block_role_probabilities: torch.Tensor | None = None
    resident_blocks: torch.Tensor | None = None
    hardware_profile_id: str | None = None
    cost_model_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_tensor("query_centroids", self.query_centroids, 4)
        _require_tensor("query_group_sizes", self.query_group_sizes, 3)
        _require_tensor("key_prototypes", self.key_prototypes, 4)
        _require_tensor("value_prototypes", self.value_prototypes, 4)
        if self.key_prototypes.shape != self.value_prototypes.shape:
            raise ValueError("key/value prototypes must have identical shape")
        if self.query_centroids.shape[:2] != self.key_prototypes.shape[:2]:
            raise ValueError("query and KV prototypes must share batch/head axes")
        blocks = self.key_prototypes.shape[2]
        for name in (
            "block_frame_ids",
            "block_token_starts",
            "block_token_ends",
            "block_age",
        ):
            value = getattr(self, name)
            _require_tensor(name, value, 1)
            if value.shape[0] != blocks:
                raise ValueError(f"{name} must have one value per history block")
        if bool((self.block_token_ends <= self.block_token_starts).any()):
            raise ValueError("history blocks must have positive width")
        if self.past_attention_score is not None:
            if self.past_attention_score.shape[-1] != blocks:
                raise ValueError("past_attention_score must end in history blocks")
        if self.resident_blocks is not None:
            if self.resident_blocks.dtype != torch.bool:
                raise ValueError("resident_blocks must be boolean")
            if self.resident_blocks.shape[-1] != blocks:
                raise ValueError("resident_blocks must end in history blocks")

    @property
    def blocks(self) -> int:
        return int(self.key_prototypes.shape[2])

    def as_dict(self) -> dict[str, Any]:
        def shape(value: torch.Tensor | None) -> list[int] | None:
            return list(value.shape) if value is not None else None

        return {
            "query_centroids_shape": shape(self.query_centroids),
            "query_group_sizes_shape": shape(self.query_group_sizes),
            "key_prototypes_shape": shape(self.key_prototypes),
            "value_prototypes_shape": shape(self.value_prototypes),
            "blocks": self.blocks,
            "past_attention_score_shape": shape(self.past_attention_score),
            "query_role_probabilities_shape": shape(
                self.query_role_probabilities
            ),
            "block_role_probabilities_shape": shape(
                self.block_role_probabilities
            ),
            "resident_blocks_shape": shape(self.resident_blocks),
            "hardware_profile_id": self.hardware_profile_id,
            "cost_model_version": self.cost_model_version,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class OfflineTeacherContext:
    """Full tensors allowed only in isolated calibration and audit scripts."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    dense_output: torch.Tensor | None = None
    dense_attention: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_tensor("query", self.query, 4)
        _require_tensor("key", self.key, 4)
        _require_tensor("value", self.value, 4)
        if self.key.shape != self.value.shape:
            raise ValueError("teacher key/value tensors must share shape")
        if self.query.shape[0] != self.key.shape[0] or self.query.shape[2:] != self.key.shape[2:]:
            raise ValueError("teacher Q/K/V batch, head and dimension axes must match")
        if self.dense_output is not None and self.dense_output.shape != self.query.shape:
            raise ValueError("dense_output must match query shape")
