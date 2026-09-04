"""Denoising, query, chunk-lifetime and layer route-reuse measurements."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch

from .history_cache import tensor_sha256
from .route_plan import HistoryRoutePlan


def route_coordinate_set(plan: HistoryRoutePlan) -> set[tuple[int, int, int, int]]:
    frames = plan.union_frame_ids.detach().to("cpu")
    tokens = plan.union_token_ids.detach().to("cpu")
    result = set()
    for batch in range(frames.shape[0]):
        for head in range(frames.shape[1]):
            for index in range(frames.shape[2]):
                frame = int(frames[batch, head, index])
                token = int(tokens[batch, head, index])
                if frame >= 0 and token >= 0:
                    result.add((batch, head, frame, token))
    return result


def set_jaccard(left: Iterable[Any], right: Iterable[Any]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


@dataclass(frozen=True)
class RouteReuseObservation:
    layer_id: int
    chunk_id: int
    denoising_pass: int
    route_plan_sha256: str
    coordinate_sha256: str
    materialized_kv_sha256: str | None = None
    rope_position_sha256: str | None = None


class RouteReuseTracker:
    def __init__(self) -> None:
        self.records: list[tuple[RouteReuseObservation, set[tuple[int, int, int, int]]]] = []

    def record(
        self,
        plan: HistoryRoutePlan,
        *,
        layer_id: int,
        chunk_id: int,
        denoising_pass: int,
        materialized_kv: torch.Tensor | None = None,
        rope_positions: torch.Tensor | None = None,
    ) -> RouteReuseObservation:
        coordinates = route_coordinate_set(plan)
        encoded = torch.tensor(sorted(coordinates), dtype=torch.long) if coordinates else torch.empty((0, 4), dtype=torch.long)
        observation = RouteReuseObservation(
            layer_id=int(layer_id),
            chunk_id=int(chunk_id),
            denoising_pass=int(denoising_pass),
            route_plan_sha256=plan.digest(),
            coordinate_sha256=tensor_sha256(encoded),
            materialized_kv_sha256=(tensor_sha256(materialized_kv) if materialized_kv is not None else None),
            rope_position_sha256=(tensor_sha256(rope_positions) if rope_positions is not None else None),
        )
        self.records.append((observation, coordinates))
        return observation

    def denoising_summary(self, *, layer_id: int, chunk_id: int) -> dict[str, Any]:
        selected = [
            item for item in self.records
            if item[0].layer_id == layer_id and item[0].chunk_id == chunk_id
        ]
        selected.sort(key=lambda item: item[0].denoising_pass)
        if not selected:
            raise KeyError("no reuse observations for layer/chunk")
        first_observation, first_coordinates = selected[0]
        rows = []
        for observation, coordinates in selected:
            rows.append(
                {
                    **asdict(observation),
                    "jaccard_vs_first": set_jaccard(first_coordinates, coordinates),
                    "same_route_sha_as_first": observation.route_plan_sha256 == first_observation.route_plan_sha256,
                    "same_materialized_kv_as_first": observation.materialized_kv_sha256 == first_observation.materialized_kv_sha256,
                    "same_rope_positions_as_first": observation.rope_position_sha256 == first_observation.rope_position_sha256,
                }
            )
        return {
            "layer_id": layer_id,
            "chunk_id": chunk_id,
            "passes": len(rows),
            "min_jaccard_vs_first": min(row["jaccard_vs_first"] for row in rows),
            "records": rows,
        }
