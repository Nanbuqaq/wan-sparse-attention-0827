"""Strictly keyed GPU cache for immutable archived history K/V."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class HistoryKVCacheKey:
    layer_id: int
    archive_epoch: int
    storage_version: int
    current_frame_id: int
    candidate_frame_ids: tuple[int, ...]
    selected_coordinate_sha256: str
    route_plan_sha256: str
    rope_policy: str
    rope_position_sha256: str
    dtype: str
    device: str
    transfer_layout: str
    padding_strategy: str

    def __post_init__(self) -> None:
        for name in (
            "layer_id",
            "archive_epoch",
            "storage_version",
            "current_frame_id",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not self.candidate_frame_ids:
            raise ValueError("candidate_frame_ids cannot be empty")
        for name in (
            "selected_coordinate_sha256",
            "route_plan_sha256",
            "rope_position_sha256",
        ):
            if len(str(getattr(self, name))) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest")

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()


@dataclass
class CachedHistoryKV:
    key: HistoryKVCacheKey
    value: torch.Tensor
    key_unrotated: torch.Tensor
    key_roped: torch.Tensor | None
    positions: torch.Tensor
    transfer_plan_sha256: str | None = None

    @property
    def bytes(self) -> int:
        tensors = [self.value, self.key_unrotated, self.positions]
        if self.key_roped is not None and self.key_roped.data_ptr() != self.key_unrotated.data_ptr():
            tensors.append(self.key_roped)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


class HistoryUnionCache:
    """LRU storage whose entries are valid only for an exact semantic key."""

    def __init__(self, budget_bytes: int):
        if budget_bytes < 0:
            raise ValueError("budget_bytes must be non-negative")
        self.budget_bytes = int(budget_bytes)
        self._entries: OrderedDict[HistoryKVCacheKey, CachedHistoryKV] = OrderedDict()
        self.current_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.active_chunk: int | None = None

    def get(self, key: HistoryKVCacheKey) -> CachedHistoryKV | None:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return entry

    def put(self, entry: CachedHistoryKV) -> None:
        if self.budget_bytes == 0:
            raise MemoryError("history union cache is disabled by a zero-byte budget")
        size = entry.bytes
        if size > self.budget_bytes:
            raise MemoryError(
                f"history union entry {size} bytes exceeds cache budget {self.budget_bytes}"
            )
        existing = self._entries.pop(entry.key, None)
        if existing is not None:
            self.current_bytes -= existing.bytes
        while self._entries and self.current_bytes + size > self.budget_bytes:
            _, removed = self._entries.popitem(last=False)
            self.current_bytes -= removed.bytes
            self.evictions += 1
        self._entries[entry.key] = entry
        self.current_bytes += size

    def clear(self) -> None:
        self._entries.clear()
        self.current_bytes = 0

    def reset(self) -> None:
        self.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.active_chunk = None

    def begin_chunk(self, current_frame_id: int, *, per_chunk: bool) -> None:
        current_frame_id = int(current_frame_id)
        if per_chunk and self.active_chunk != current_frame_id:
            self.clear()
        self.active_chunk = current_frame_id

    def clear_chunk(self, current_frame_id: int) -> None:
        retained = OrderedDict(
            (key, entry)
            for key, entry in self._entries.items()
            if key.current_frame_id == int(current_frame_id)
        )
        self._entries = retained
        self.current_bytes = sum(entry.bytes for entry in retained.values())

    def as_dict(self) -> dict[str, Any]:
        accesses = self.hits + self.misses
        return {
            "budget_bytes": self.budget_bytes,
            "current_bytes": self.current_bytes,
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "active_chunk": self.active_chunk,
            "hit_rate": self.hits / accesses if accesses else None,
        }


def tensor_sha256(value: torch.Tensor) -> str:
    cpu = value.detach().to("cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode())
    digest.update(json.dumps(list(cpu.shape)).encode())
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()
