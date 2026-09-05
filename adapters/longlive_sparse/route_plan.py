"""Backend-independent autoregressive history routing plan."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import torch


def map_union_coordinates(
    route_plan: "HistoryRoutePlan",
    candidate_frame_ids: torch.Tensor,
    candidate_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Map route-plan coordinates into the original dense candidate order."""

    device = candidate_frame_ids.device
    frame_ids = route_plan.union_frame_ids.to(device)
    token_ids = route_plan.union_token_ids.to(device)
    valid = frame_ids >= 0
    max_token = max(
        int(candidate_token_ids.max()) if candidate_token_ids.numel() else 0,
        int(token_ids[valid].max()) if valid.any() else 0,
    )
    base = max_token + 1
    candidate_codes = candidate_frame_ids.long() * base + candidate_token_ids.long()
    union_codes = frame_ids.long() * base + token_ids.clamp_min(0).long()
    sorted_codes, sorted_to_dense = torch.sort(candidate_codes, dim=-1)
    sorted_indices = torch.searchsorted(
        sorted_codes.contiguous(), union_codes.contiguous()
    ).clamp_max(candidate_codes.shape[-1] - 1)
    indices = sorted_to_dense.gather(-1, sorted_indices)
    matched = candidate_codes.gather(-1, indices) == union_codes
    if not bool((matched | ~valid).all()):
        raise KeyError("route plan contains coordinates outside the dense candidate transfer")
    return torch.where(valid, indices, torch.zeros_like(indices))


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
        if 'routing_identity' in self.metadata:
            digest.update(json.dumps(self.metadata['routing_identity'], sort_keys=True, separators=(',', ':')).encode())
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

    def state_dict(self) -> dict[str, Any]:
        """Return a portable CPU representation for exact backend replay."""

        return {
            "method": self.method,
            "routing_stage": self.routing_stage,
            "query_labels": self.query_labels.detach().to("cpu"),
            "query_group_sizes": self.query_group_sizes.detach().to("cpu"),
            "union_frame_ids": self.union_frame_ids.detach().to("cpu"),
            "union_token_ids": self.union_token_ids.detach().to("cpu"),
            "group_union_indices": self.group_union_indices.detach().to("cpu"),
            "group_history_counts": self.group_history_counts.detach().to("cpu"),
            "candidate_history_tokens": self.candidate_history_tokens,
            "query_tokens": self.query_tokens,
            "exact_k_tokens": self.exact_k_tokens,
            "target_history_density": self.target_history_density,
            "backend_hint": self.backend_hint,
            "metadata": self.metadata,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "HistoryRoutePlan":
        """Restore a route plan without recomputing routing decisions."""

        return cls(
            method=str(state["method"]),
            routing_stage=str(state["routing_stage"]),
            query_labels=state["query_labels"],
            query_group_sizes=state["query_group_sizes"],
            union_frame_ids=state["union_frame_ids"],
            union_token_ids=state["union_token_ids"],
            group_union_indices=state["group_union_indices"],
            group_history_counts=state["group_history_counts"],
            candidate_history_tokens=int(state["candidate_history_tokens"]),
            query_tokens=int(state["query_tokens"]),
            exact_k_tokens=int(state["exact_k_tokens"]),
            target_history_density=float(state["target_history_density"]),
            backend_hint=str(state.get("backend_hint", "grouped_fa2")),
            metadata=dict(state.get("metadata", {})),
        )

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
