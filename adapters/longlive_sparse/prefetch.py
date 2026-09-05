"""Prediction, bounded admission, validation and completion for KV prefetch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Hashable, Iterable


BlockId = Hashable


def _ordered_unique(values: Iterable[BlockId]) -> list[BlockId]:
    result = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


@dataclass(frozen=True)
class VerifiedPrefetchPlan:
    predicted: tuple[BlockId, ...]
    admitted_predictions: tuple[BlockId, ...]
    resident_hits: tuple[BlockId, ...]
    prediction_hits: tuple[BlockId, ...]
    completion_misses: tuple[BlockId, ...]
    extras: tuple[BlockId, ...]
    actual: tuple[BlockId, ...]
    ready_before_use: tuple[BlockId, ...]
    bytes_per_block: int

    @property
    def prediction_recall(self) -> float:
        return len(set(self.predicted) & set(self.actual)) / len(self.actual) if self.actual else 1.0

    @property
    def admitted_recall(self) -> float:
        return len(self.prediction_hits) / len(self.actual) if self.actual else 1.0

    @property
    def prediction_precision(self) -> float:
        return len(set(self.predicted) & set(self.actual)) / len(self.predicted) if self.predicted else 1.0

    @property
    def admitted_precision(self) -> float:
        return len(self.prediction_hits) / len(self.admitted_predictions) if self.admitted_predictions else 1.0

    @property
    def timeliness(self) -> float:
        useful = set(self.prediction_hits)
        return len(useful & set(self.ready_before_use)) / len(useful) if useful else 1.0

    @property
    def extra_bytes(self) -> int:
        return len(self.extras) * self.bytes_per_block

    @property
    def miss_bytes(self) -> int:
        return len(self.completion_misses) * self.bytes_per_block

    @property
    def newly_admitted_bytes(self) -> int:
        return len(self.admitted_predictions) * self.bytes_per_block

    def final_execution_blocks(self) -> tuple[BlockId, ...]:
        """The verified path always executes exactly the actual route set."""

        return self.actual

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload.update(
            {
                "prediction_recall": self.prediction_recall,
                "admitted_recall": self.admitted_recall,
                "prediction_precision": self.prediction_precision,
                "admitted_precision": self.admitted_precision,
                "timeliness": self.timeliness,
                "extra_bytes": self.extra_bytes,
                "miss_bytes": self.miss_bytes,
                "newly_admitted_bytes": self.newly_admitted_bytes,
            }
        )
        return payload


def build_verified_prefetch_plan(
    predicted: Iterable[BlockId],
    actual: Iterable[BlockId],
    resident: Iterable[BlockId],
    *,
    bytes_per_block: int,
    max_new_blocks: int,
    max_new_bytes: int,
    ready_before_use: Iterable[BlockId] = (),
) -> VerifiedPrefetchPlan:
    if bytes_per_block <= 0:
        raise ValueError("bytes_per_block must be positive")
    if max_new_blocks < 0 or max_new_bytes < 0:
        raise ValueError("prefetch caps must be non-negative")
    predicted_order = _ordered_unique(predicted)
    actual_order = _ordered_unique(actual)
    resident_set = set(resident)
    admission_limit = min(max_new_blocks, max_new_bytes // bytes_per_block)
    admitted = []
    for block in predicted_order:
        if block in resident_set:
            continue
        if len(admitted) >= admission_limit:
            break
        admitted.append(block)
    actual_set = set(actual_order)
    admitted_set = set(admitted)
    resident_hits = [block for block in actual_order if block in resident_set]
    prediction_hits = [block for block in actual_order if block in admitted_set]
    misses = [
        block
        for block in actual_order
        if block not in resident_set and block not in admitted_set
    ]
    extras = [block for block in admitted if block not in actual_set]
    ready = [block for block in _ordered_unique(ready_before_use) if block in admitted_set]
    return VerifiedPrefetchPlan(
        predicted=tuple(predicted_order),
        admitted_predictions=tuple(admitted),
        resident_hits=tuple(resident_hits),
        prediction_hits=tuple(prediction_hits),
        completion_misses=tuple(misses),
        extras=tuple(extras),
        actual=tuple(actual_order),
        ready_before_use=tuple(ready),
        bytes_per_block=bytes_per_block,
    )
