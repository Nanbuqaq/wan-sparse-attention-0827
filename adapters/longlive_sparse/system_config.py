"""System-level configuration for LongLive history KV movement and execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


PROFILE_MODES = {"off", "summary", "trace"}
TRANSFER_LAYOUTS = {"legacy", "exact_compact", "block64", "page256", "frame1560"}
CACHE_MODES = {"off", "per_chunk", "cross_chunk"}
CACHE_PAYLOADS = {"raw_kv", "roped_kv"}
OFFLOAD_OVERLAPS = {"none", "d2h_compute"}
ONLOAD_OVERLAPS = {"none", "kv_stream"}
EXECUTION_DATAFLOWS = {
    "qout_grouped_fa2",
    "biased_sdpa_reference",
    "kvout_online",
}
GROUP_SELECTION_POLICIES = {"legacy_exact_union", "mass_preserving_top_p"}
STAGING_MODES = {
    "per_call_separate",
    "persistent_separate",
    "persistent_fused",
}


@dataclass(frozen=True)
class LongLiveSystemConfig:
    """Decisions that affect physical KV movement but not method identity.

    Defaults preserve the historical ``longlive-sparse`` behavior.  Every
    non-default experiment must serialize this object into its case identity.
    """

    profile_mode: str = "off"
    transfer_layout: str = "legacy"
    gpu_union_cache: str = "off"
    gpu_union_cache_budget_mib: int = 0
    cache_payload: str = "roped_kv"
    offload_overlap: str = "none"
    onload_overlap: str = "none"
    execution_dataflow: str = "qout_grouped_fa2"
    pinned_buffer_slots: int = 2
    host_pinned_budget_mib: int = 1024
    staging_mode: str = "per_call_separate"
    cpu_pack_policy: str = "candidate_gather"
    page_tokens: int = 256
    group_selection_policy: str = "legacy_exact_union"
    group_top_p: float = 0.90
    group_min_k_ratio: float = 0.10
    hardware_profile_id: str | None = None
    cost_model_version: str | None = None

    def __post_init__(self) -> None:
        if self.profile_mode not in PROFILE_MODES:
            raise ValueError(f"unsupported profile_mode: {self.profile_mode!r}")
        if self.transfer_layout not in TRANSFER_LAYOUTS:
            raise ValueError(f"unsupported transfer_layout: {self.transfer_layout!r}")
        if self.gpu_union_cache not in CACHE_MODES:
            raise ValueError(f"unsupported gpu_union_cache: {self.gpu_union_cache!r}")
        if self.gpu_union_cache_budget_mib < 0:
            raise ValueError("gpu_union_cache_budget_mib must be non-negative")
        if self.cache_payload not in CACHE_PAYLOADS:
            raise ValueError(f"unsupported cache_payload: {self.cache_payload!r}")
        if self.gpu_union_cache == "cross_chunk" and self.cache_payload != "raw_kv":
            raise ValueError("cross_chunk cache requires raw_kv and re-RoPE on consumption")
        if self.offload_overlap not in OFFLOAD_OVERLAPS:
            raise ValueError(f"unsupported offload_overlap: {self.offload_overlap!r}")
        if self.onload_overlap not in ONLOAD_OVERLAPS:
            raise ValueError(f"unsupported onload_overlap: {self.onload_overlap!r}")
        if self.execution_dataflow not in EXECUTION_DATAFLOWS:
            raise ValueError(
                f"unsupported execution_dataflow: {self.execution_dataflow!r}"
            )
        if self.pinned_buffer_slots < 1:
            raise ValueError("pinned_buffer_slots must be positive")
        if self.host_pinned_budget_mib < 0:
            raise ValueError("host_pinned_budget_mib must be non-negative")
        if self.staging_mode not in STAGING_MODES:
            raise ValueError(f"unsupported staging_mode: {self.staging_mode!r}")
        if self.cpu_pack_policy not in {"candidate_gather", "archive_runs"}:
            raise ValueError('unsupported cpu_pack_policy')
        if self.page_tokens < 1:
            raise ValueError("page_tokens must be positive")
        if self.group_selection_policy not in GROUP_SELECTION_POLICIES:
            raise ValueError(
                "unsupported group_selection_policy: "
                f"{self.group_selection_policy!r}"
            )
        if not 0.0 < self.group_top_p <= 1.0:
            raise ValueError("group_top_p must be in (0, 1]")
        if not 0.0 <= self.group_min_k_ratio <= 1.0:
            raise ValueError("group_min_k_ratio must be in [0, 1]")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def identity_dict(self) -> dict[str, Any]:
        """Return all fields that must distinguish a system experiment."""

        return self.as_dict()

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None
    ) -> "LongLiveSystemConfig":
        return cls(**dict(value or {}))
