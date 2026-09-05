"""Reusable CPU staging buffers for separate or fused K/V copies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class StagingLease:
    slot: int
    key: torch.Tensor
    value: torch.Tensor
    fused: torch.Tensor | None
    reused: bool

    @property
    def bytes(self) -> int:
        if self.fused is not None:
            return self.fused.numel() * self.fused.element_size()
        return (self.key.numel() + self.value.numel()) * self.key.element_size()


class PinnedStagingPool:
    def __init__(self, *, slots: int, budget_bytes: int, pin_memory: bool):
        if slots < 1 or budget_bytes < 0:
            raise ValueError("staging slots must be positive and budget non-negative")
        self.slots = int(slots)
        self.budget_bytes = int(budget_bytes)
        self.pin_memory = bool(pin_memory)
        self._buffers: list[dict[str, Any] | None] = [None] * slots
        self._in_use = [False] * slots
        self.allocations = 0
        self.reuses = 0
        self.peak_bytes = 0

    def _allocated_bytes(self) -> int:
        total = 0
        for item in self._buffers:
            if item is not None:
                total += int(item["bytes"])
        return total

    def acquire(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        fused: bool,
    ) -> StagingLease:
        required = 2 * int(torch.tensor(shape).prod()) * torch.empty((), dtype=dtype).element_size()
        for slot, item in enumerate(self._buffers):
            if self._in_use[slot]:
                continue
            if (
                item is not None
                and item["shape"] == shape
                and item["dtype"] == dtype
                and item["fused_mode"] == fused
            ):
                self._in_use[slot] = True
                self.reuses += 1
                return StagingLease(
                    slot=slot,
                    key=item["key"],
                    value=item["value"],
                    fused=item["fused"],
                    reused=True,
                )
        slot = next((index for index, used in enumerate(self._in_use) if not used), None)
        if slot is None:
            raise RuntimeError("all staging slots are in use")
        existing = self._buffers[slot]
        existing_bytes = int(existing["bytes"]) if existing is not None else 0
        projected = self._allocated_bytes() - existing_bytes + required
        if projected > self.budget_bytes:
            raise MemoryError(
                f"staging allocation {projected} exceeds budget {self.budget_bytes}"
            )
        if fused:
            buffer = torch.empty(
                (2, *shape), dtype=dtype, device="cpu", pin_memory=self.pin_memory
            )
            key, value = buffer[0], buffer[1]
        else:
            buffer = None
            key = torch.empty(
                shape, dtype=dtype, device="cpu", pin_memory=self.pin_memory
            )
            value = torch.empty_like(key, pin_memory=self.pin_memory)
        self._buffers[slot] = {
            "shape": shape,
            "dtype": dtype,
            "fused_mode": fused,
            "fused": buffer,
            "key": key,
            "value": value,
            "bytes": required,
        }
        self._in_use[slot] = True
        self.allocations += 1
        self.peak_bytes = max(self.peak_bytes, projected)
        return StagingLease(slot, key, value, buffer, False)

    def release(self, lease: StagingLease) -> None:
        if not 0 <= lease.slot < self.slots or not self._in_use[lease.slot]:
            raise ValueError("staging lease is not active")
        self._in_use[lease.slot] = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "slots": self.slots,
            "budget_bytes": self.budget_bytes,
            "pin_memory": self.pin_memory,
            "allocated_bytes": self._allocated_bytes(),
            "peak_bytes": self.peak_bytes,
            "allocations": self.allocations,
            "reuses": self.reuses,
        }
