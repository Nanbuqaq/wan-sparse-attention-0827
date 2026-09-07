"""Exact-value-validated CPU route identity reuse; never caches Q/K/V.

Only the current route per attention module is retained. Archive epochs,
storage versions, dtype/device/layout still enter the original KV cache key.
"""
from dataclasses import dataclass

import torch

from .history_cache import tensor_sha256
from .rope import build_sparse_positions


@dataclass(frozen=True)
class PreparedRouteIdentity:
    route_sha256: str
    coordinate_sha256: str
    position_sha256: str
    positions: torch.Tensor


class RouteIdentityCache:
    def __init__(self):
        self._plan = None
        self._key = None
        self._value = None
        self.hits = self.misses = 0

    def clear(self):
        if self._plan is not None:
            self._plan.disable_verified_digest_reuse()
        self._plan = self._key = self._value = None

    def prepare(self, plan, *, current_frame_id, spatial_width, rope_policy,
                max_relative_age, candidate_frame_ids):
        if self._plan is not plan:
            self.clear()
            self._plan = plan
            plan.enable_verified_digest_reuse()
        route_sha = plan.digest()  # Exact comparisons invalidate alias mutations.
        candidates = tuple(int(x) for x in candidate_frame_ids)
        identity = (route_sha, int(current_frame_id), int(spatial_width), str(rope_policy),
                    int(max_relative_age), candidates)
        if self._key == identity:
            self.hits += 1
            return self._value
        positions = build_sparse_positions(frame_ids=plan.union_frame_ids.clamp_min(0),
            token_ids=plan.union_token_ids.clamp_min(0), current_frame_id=current_frame_id,
            spatial_width=spatial_width, rope_policy=rope_policy, max_relative_age=max_relative_age,
            candidate_frame_ids=torch.tensor(candidates, dtype=torch.long))
        coordinates = torch.stack((plan.union_frame_ids.long(), plan.union_token_ids.long()), dim=-1)
        value = PreparedRouteIdentity(route_sha, tensor_sha256(coordinates), tensor_sha256(positions), positions)
        self._key, self._value = identity, value
        self.misses += 1
        return value

    @property
    def retained_CPU_bytes(self):
        if self._value is None:
            return 0
        return self._plan.digest_snapshot_bytes + self._value.positions.numel() * self._value.positions.element_size()
