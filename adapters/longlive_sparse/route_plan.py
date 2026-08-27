"""Backend-independent autoregressive history routing plan."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class HistoryRoutePlan:
    method: str
    routing_stage: str
    query_labels: torch.Tensor
    query_group_sizes: torch.Tensor
    union_frame_ids: torch.Tensor
    union_token_ids: torch.Tensor
    group_union_indices: torch.Tensor
    group_history_counts: torch.Tensor
    candidate_history_tokens: int
    query_tokens: int
    exact_k_tokens: int
    target_history_density: float
    backend_hint: str = "grouped_fa2"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def groups(self) -> int:
        return int(self.query_group_sizes.shape[-1])

    @property
    def unique_history_tokens(self) -> int:
        return int((self.union_frame_ids >= 0).sum())

    @property
    def history_pairs(self) -> int:
        return int((self.query_group_sizes.long() * self.group_history_counts.long()).sum())

    @property
    def full_history_pairs(self) -> int:
        batch, heads = self.query_labels.shape[:2]
        return int(batch * heads * self.query_tokens * self.candidate_history_tokens)

    @property
    def history_pair_density(self) -> float:
        return self.history_pairs / self.full_history_pairs if self.full_history_pairs else 1.0

    @property
    def history_transfer_density(self) -> float:
        denominator = self.candidate_history_tokens * self.union_frame_ids.shape[0] * self.union_frame_ids.shape[1]
        return self.unique_history_tokens / denominator if denominator else 1.0

    @property
    def global_executed_density(self) -> float:
        batch, heads = self.query_labels.shape[:2]
        exact_pairs = batch * heads * self.query_tokens * self.exact_k_tokens
        dense_pairs = exact_pairs + self.full_history_pairs
        return (exact_pairs + self.history_pairs) / dense_pairs if dense_pairs else 1.0

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.method.encode())
        digest.update(self.routing_stage.encode())
        for tensor in (
            self.query_labels,
            self.query_group_sizes,
            self.union_frame_ids,
            self.union_token_ids,
            self.group_union_indices,
            self.group_history_counts,
        ):
            cpu = tensor.detach().to("cpu").contiguous()
            digest.update(str(cpu.dtype).encode())
            digest.update(json.dumps(list(cpu.shape)).encode())
            digest.update(cpu.numpy().tobytes())
        return digest.hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "routing_stage": self.routing_stage,
            "groups": self.groups,
            "candidate_history_tokens": self.candidate_history_tokens,
            "unique_history_tokens": self.unique_history_tokens,
            "query_tokens": self.query_tokens,
            "exact_k_tokens": self.exact_k_tokens,
            "target_history_density": self.target_history_density,
            "history_pair_density": self.history_pair_density,
            "history_transfer_density": self.history_transfer_density,
            "global_executed_density": self.global_executed_density,
            "route_plan_sha256": self.digest(),
            "backend_hint": self.backend_hint,
            "metadata": self.metadata,
        }
