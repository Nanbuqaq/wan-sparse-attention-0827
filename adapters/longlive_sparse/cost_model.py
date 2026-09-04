"""Frozen predictive system-cost model, separate from physical transfer plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .route_plan import HistoryRoutePlan
from .transfer_plan import TransferPlan


@dataclass(frozen=True)
class HardwareCostProfile:
    profile_id: str
    model_version: str
    h2d_bytes_per_second: float
    hbm_bytes_per_second: float
    copy_launch_seconds: float
    pack_run_seconds: float
    source_artifact_sha256: str

    def __post_init__(self) -> None:
        if not self.profile_id or not self.model_version:
            raise ValueError("hardware cost profile identity cannot be empty")
        for name in (
            "h2d_bytes_per_second",
            "hbm_bytes_per_second",
            "copy_launch_seconds",
            "pack_run_seconds",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if len(self.source_artifact_sha256) != 64:
            raise ValueError("source_artifact_sha256 must be a SHA-256 digest")


@dataclass(frozen=True)
class CausalPipelineState:
    predicted_overlap_fraction: float = 0.0
    in_flight_copy_bytes: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.predicted_overlap_fraction <= 1.0:
            raise ValueError("predicted_overlap_fraction must be in [0,1]")
        if self.in_flight_copy_bytes < 0:
            raise ValueError("in_flight_copy_bytes must be non-negative")


@dataclass(frozen=True)
class PredictedCostBreakdown:
    profile_id: str
    model_version: str
    execution_dataflow: str
    pack_service_s: float
    h2d_service_s: float
    hbm_service_s: float
    predicted_exposed_h2d_s: float
    predicted_exposed_wait_s: float
    physical_copy_bytes: int
    hbm_bytes: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SystemCostModel:
    """A simple, auditable model calibrated only from isolated profiles."""

    def __init__(self, profile: HardwareCostProfile):
        self.profile = profile

    def predict(
        self,
        route_plan: HistoryRoutePlan,
        transfer_plan: TransferPlan,
        *,
        execution_dataflow: str,
        pipeline_state: CausalPipelineState | None = None,
        query_reuse_factor: float = 1.0,
    ) -> PredictedCostBreakdown:
        if transfer_plan.route_plan_sha256 != route_plan.digest():
            raise ValueError("transfer plan does not belong to the supplied route plan")
        if query_reuse_factor < 1.0:
            raise ValueError("query_reuse_factor must be at least one")
        state = pipeline_state or CausalPipelineState()
        pack_service = (
            transfer_plan.source_run_count * self.profile.pack_run_seconds
        )
        h2d_service = (
            transfer_plan.physical_copy_bytes / self.profile.h2d_bytes_per_second
            + transfer_plan.source_run_count * self.profile.copy_launch_seconds
        )
        if execution_dataflow == "qout_grouped_fa2":
            hbm_multiplier = query_reuse_factor
        elif execution_dataflow == "kvout_online":
            hbm_multiplier = 1.0
        elif execution_dataflow == "biased_sdpa_reference":
            hbm_multiplier = max(2.0, query_reuse_factor)
        else:
            raise ValueError(f"unsupported execution_dataflow: {execution_dataflow!r}")
        hbm_bytes = float(transfer_plan.logical_tokens * transfer_plan.bytes_per_token) * hbm_multiplier
        hbm_service = hbm_bytes / self.profile.hbm_bytes_per_second
        exposed_h2d = h2d_service * (1.0 - state.predicted_overlap_fraction)
        predicted_wait = pack_service + exposed_h2d + hbm_service
        return PredictedCostBreakdown(
            profile_id=self.profile.profile_id,
            model_version=self.profile.model_version,
            execution_dataflow=execution_dataflow,
            pack_service_s=pack_service,
            h2d_service_s=h2d_service,
            hbm_service_s=hbm_service,
            predicted_exposed_h2d_s=exposed_h2d,
            predicted_exposed_wait_s=predicted_wait,
            physical_copy_bytes=transfer_plan.physical_copy_bytes,
            hbm_bytes=hbm_bytes,
        )


def mean_absolute_percentage_error(
    predicted: Sequence[float], measured: Sequence[float]
) -> float:
    if len(predicted) != len(measured) or not predicted:
        raise ValueError("predicted and measured must be non-empty equal-length sequences")
    values = []
    for estimate, truth in zip(predicted, measured):
        if truth <= 0:
            raise ValueError("measured costs must be positive")
        values.append(abs(float(estimate) - float(truth)) / float(truth))
    return sum(values) / len(values)
