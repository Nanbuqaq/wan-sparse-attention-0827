"""Timeline utilities that keep service time separate from exposed wait."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TimelineInterval:
    name: str
    start_s: float
    end_s: float
    resource: str

    def __post_init__(self) -> None:
        if self.start_s < 0 or self.end_s < self.start_s:
            raise ValueError("invalid timeline interval")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def interval_union_duration(intervals: Iterable[TimelineInterval]) -> float:
    spans = sorted((item.start_s, item.end_s) for item in intervals)
    if not spans:
        return 0.0
    total = 0.0
    start, end = spans[0]
    for next_start, next_end in spans[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def overlap_duration(
    left: Iterable[TimelineInterval], right: Iterable[TimelineInterval]
) -> float:
    intersections = []
    for first in left:
        for second in right:
            start = max(first.start_s, second.start_s)
            end = min(first.end_s, second.end_s)
            if end > start:
                intersections.append(TimelineInterval("intersection", start, end, "derived"))
    return interval_union_duration(intersections)
