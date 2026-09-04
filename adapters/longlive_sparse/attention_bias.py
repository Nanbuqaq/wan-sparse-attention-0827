"""Compact attention-bias plans for role- and age-aware routing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class AttentionBiasPlan:
    """A compact role representation; never a materialized Q-by-K matrix."""

    role_names: tuple[str, ...]
    query_role_probabilities: torch.Tensor
    history_role_probabilities: torch.Tensor
    history_age_weights: torch.Tensor
    mode: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.role_names:
            raise ValueError("role_names cannot be empty")
        if self.query_role_probabilities.ndim != 3:
            raise ValueError("query roles must be [B,Q,R]")
        if self.history_role_probabilities.ndim != 4:
            raise ValueError("history roles must be [B,H,U,R]")
        if self.history_age_weights.ndim != 3:
            raise ValueError("history_age_weights must be [B,H,U]")
        roles = len(self.role_names)
        if self.query_role_probabilities.shape[-1] != roles:
            raise ValueError("query role width does not match role_names")
        if self.history_role_probabilities.shape[-1] != roles:
            raise ValueError("history role width does not match role_names")
        if self.history_role_probabilities.shape[:-1] != self.history_age_weights.shape:
            raise ValueError("history role and age tensors must share B/H/U axes")
        for name, value in (
            ("query_role_probabilities", self.query_role_probabilities),
            ("history_role_probabilities", self.history_role_probabilities),
        ):
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
            if bool(((value < 0) | (value > 1)).any()):
                raise ValueError(f"{name} must be probabilities in [0,1]")
            totals = value.float().sum(dim=-1)
            if not torch.allclose(totals, torch.ones_like(totals), atol=1e-5, rtol=1e-5):
                raise ValueError(f"{name} probabilities must sum to one")
        if not torch.isfinite(self.history_age_weights).all():
            raise ValueError("history_age_weights contains non-finite values")
        if bool((self.history_age_weights <= 0).any()):
            raise ValueError("history_age_weights must be positive")

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.mode.encode())
        digest.update(json.dumps(self.role_names).encode())
        for value in (
            self.query_role_probabilities,
            self.history_role_probabilities,
            self.history_age_weights,
        ):
            cpu = value.detach().to("cpu").contiguous()
            digest.update(str(cpu.dtype).encode())
            digest.update(json.dumps(list(cpu.shape)).encode())
            digest.update(cpu.numpy().tobytes())
        return digest.hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "role_names": list(self.role_names),
            "query_shape": list(self.query_role_probabilities.shape),
            "history_shape": list(self.history_role_probabilities.shape),
            "age_shape": list(self.history_age_weights.shape),
            "attention_bias_plan_sha256": self.digest(),
            "metadata": self.metadata,
        }
