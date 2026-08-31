"""Frozen method catalogue and routing-stage contracts for LongLive AR attention."""

from __future__ import annotations

from dataclasses import asdict, dataclass


ROUTING_STAGES = {"N/A", "pre-transfer", "post-transfer", "hybrid"}
BACKENDS = {"packed_fa2", "grouped_fa2", "fixed64_rect", "varlen_triton"}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    category: str
    routing_stage: str
    counts_as_self_cluster: bool = False
    q_clusters: int = 1
    k_clusters: int = 1
    iterations: int = 5
    threshold: float | None = None
    query_threshold: float | None = None
    rank: int | None = None
    temporal_bins: int | None = None
    capacity_factor: float | None = None
    top_p: float | None = None
    fixed_topk_ratio: float | None = None
    co_cluster_iterations: int | None = None
    base_fraction: float | None = None
    local_fraction: float | None = None
    remote_clusters: int | None = None
    remote_min_frames: int | None = None
    v_weight: float | None = None
    transfer_multiplier: float | None = None
    parameter_origin: str = "initial_transfer_config"

    def __post_init__(self) -> None:
        if self.routing_stage not in ROUTING_STAGES:
            raise ValueError(f"invalid routing stage: {self.routing_stage}")
        if self.base_fraction is not None or self.local_fraction is not None:
            base = float(self.base_fraction or 0.0)
            local = float(self.local_fraction or 0.0)
            if base < 0 or local < 0 or base + local > 1:
                raise ValueError("invalid base/local budget fractions")
        if self.transfer_multiplier is not None and self.transfer_multiplier < 1.0:
            raise ValueError("transfer_multiplier must be at least 1.0")

    def as_dict(self) -> dict:
        return asdict(self)


METHOD_SPECS: dict[str, MethodSpec] = {
    "native_dense": MethodSpec("native_dense", "baseline", "N/A"),
    "native_block": MethodSpec("native_block", "baseline", "N/A"),
    "rag_dense": MethodSpec("rag_dense", "baseline", "post-transfer"),
    "dense_history": MethodSpec("dense_history", "baseline", "post-transfer"),
    "rag_local": MethodSpec("rag_local", "baseline", "pre-transfer"),
    "random_history": MethodSpec("random_history", "baseline", "pre-transfer"),
    "block64_history": MethodSpec("block64_history", "baseline", "pre-transfer"),
    "token_oracle": MethodSpec("token_oracle", "baseline", "post-transfer"),
    "kcluster32_history": MethodSpec(
        "kcluster32_history", "simple_k_cluster", "pre-transfer", k_clusters=32
    ),
    "fixed_k128_history": MethodSpec(
        "fixed_k128_history", "simple_k_cluster", "pre-transfer", k_clusters=128
    ),
    "fixed_k256_history": MethodSpec(
        "fixed_k256_history", "simple_k_cluster", "pre-transfer", k_clusters=256
    ),
    "qlocal_kmeans8_ar": MethodSpec(
        "qlocal_kmeans8_ar",
        "self_cluster",
        "pre-transfer",
        counts_as_self_cluster=True,
        q_clusters=8,
        k_clusters=25,
        iterations=3,
    ),
    "radius_k256_ar": MethodSpec(
        "radius_k256_ar",
        "self_cluster",
        "pre-transfer",
        counts_as_self_cluster=True,
        k_clusters=256,
        threshold=0.5,
    ),
    "qmetric_k256_r32_ar": MethodSpec(
        "qmetric_k256_r32_ar",
        "self_cluster",
        "hybrid",
        counts_as_self_cluster=True,
        k_clusters=256,
        rank=32,
    ),
    "temporal_k256_t16_ar": MethodSpec(
        "temporal_k256_t16_ar",
        "self_cluster",
        "pre-transfer",
        counts_as_self_cluster=True,
        k_clusters=256,
        temporal_bins=16,
    ),
    "sizesplit_k128_c2_ar": MethodSpec(
        "sizesplit_k128_c2_ar",
        "self_cluster",
        "pre-transfer",
        counts_as_self_cluster=True,
        k_clusters=128,
        capacity_factor=2.0,
    ),
    "svg2_ar": MethodSpec(
        "svg2_ar",
        "paper",
        "hybrid",
        q_clusters=300,
        k_clusters=1000,
        iterations=5,
        top_p=0.90,
        fixed_topk_ratio=0.10,
        parameter_origin="official_wan_shape_initial_longlive_calibration_pending",
    ),
    "adacluster_ar": MethodSpec(
        "adacluster_ar",
        "paper",
        "hybrid",
        q_clusters=65,
        k_clusters=100,
        threshold=5.5,
        query_threshold=9.0,
        parameter_origin="official_threshold_initial_longlive_calibration_pending",
    ),
    "svoo_ar": MethodSpec(
        "svoo_ar",
        "paper",
        "post-transfer",
        q_clusters=256,
        k_clusters=1024,
        iterations=2,
        top_p=0.90,
        co_cluster_iterations=2,
        parameter_origin="official_wan_1p3b_initial_longlive_calibration_pending",
    ),
    "scope_ar": MethodSpec(
        "scope_ar",
        "paper",
        "post-transfer",
        q_clusters=100,
        k_clusters=333,
        top_p=0.90,
        fixed_topk_ratio=0.10,
        parameter_origin="paper_probe_initial_longlive_calibration_pending",
    ),
    "coverage_cluster_history": MethodSpec(
        "coverage_cluster_history",
        "proposed",
        "pre-transfer",
        k_clusters=128,
        iterations=5,
        base_fraction=0.70,
        local_fraction=0.15,
        remote_clusters=128,
        remote_min_frames=2,
        parameter_origin="initial_70_15_15_candidate_pending_isolated_calibration",
    ),
    "vaware_cluster_history": MethodSpec(
        "vaware_cluster_history",
        "proposed",
        "pre-transfer",
        k_clusters=128,
        iterations=5,
        base_fraction=0.80,
        local_fraction=0.10,
        remote_clusters=128,
        remote_min_frames=2,
        v_weight=0.75,
        parameter_origin="initial_80_10_10_online_prototype_candidate_pending_isolated_calibration",
    ),
    "transfer_vaware_hybrid_history": MethodSpec(
        "transfer_vaware_hybrid_history",
        "proposed",
        "pre-transfer",
        k_clusters=128,
        iterations=5,
        base_fraction=0.80,
        local_fraction=0.10,
        remote_clusters=128,
        remote_min_frames=2,
        v_weight=0.75,
        transfer_multiplier=1.25,
        parameter_origin="initial_transfer_aware_candidate_pending_isolated_calibration",
    ),
}


def method_spec(name: str) -> MethodSpec:
    try:
        return METHOD_SPECS[name]
    except KeyError as error:
        raise ValueError(f"unknown LongLive sparse method: {name}") from error


def validate_method_coverage() -> None:
    self_methods = [spec for spec in METHOD_SPECS.values() if spec.counts_as_self_cluster]
    paper_methods = {spec.name for spec in METHOD_SPECS.values() if spec.category == "paper"}
    if len(self_methods) < 5:
        raise RuntimeError("method table requires at least five distinct self-clustering directions")
    if paper_methods != {"svg2_ar", "adacluster_ar", "svoo_ar", "scope_ar"}:
        raise RuntimeError(f"paper method coverage mismatch: {paper_methods}")
    proposed = {spec.name for spec in METHOD_SPECS.values() if spec.category == "proposed"}
    expected_proposed = {
        "coverage_cluster_history",
        "vaware_cluster_history",
        "transfer_vaware_hybrid_history",
    }
    if proposed != expected_proposed:
        raise RuntimeError(f"proposed method coverage mismatch: {proposed}")


validate_method_coverage()
