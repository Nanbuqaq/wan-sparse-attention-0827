from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from adapters.longlive_sparse.attention_bias import AttentionBiasPlan
from adapters.longlive_sparse.contexts import OfflineTeacherContext, OnlineRoutingContext
from adapters.longlive_sparse.cost_model import (
    CausalPipelineState,
    HardwareCostProfile,
    SystemCostModel,
    mean_absolute_percentage_error,
)
from adapters.longlive_sparse.route_plan import HistoryRoutePlan
from adapters.longlive_sparse.system_config import LongLiveSystemConfig
from adapters.longlive_sparse.system_trace import SystemTraceRecord
from adapters.longlive_sparse.transfer_plan import build_transfer_plan


def route_plan() -> HistoryRoutePlan:
    return HistoryRoutePlan(
        method="block64_history",
        routing_stage="pre-transfer",
        query_labels=torch.tensor([[[0, 0, 1, 1]]]),
        query_group_sizes=torch.tensor([[[2, 2]]]),
        union_frame_ids=torch.tensor([[[5, 5, 7, -1]]]),
        union_token_ids=torch.tensor([[[1, 2, 6, -1]]]),
        group_union_indices=torch.tensor([[[[0, 1], [1, 2]]]]),
        group_history_counts=torch.tensor([[[2, 2]]]),
        candidate_history_tokens=16,
        query_tokens=4,
        exact_k_tokens=8,
        target_history_density=0.25,
    )


def test_system_config_defaults_preserve_legacy_behavior() -> None:
    config = LongLiveSystemConfig()
    assert config.transfer_layout == "legacy"
    assert config.gpu_union_cache == "off"
    assert config.offload_overlap == "none"
    assert config.onload_overlap == "none"
    assert LongLiveSystemConfig.from_mapping(config.as_dict()) == config


def test_system_config_allows_two_overlap_axes_together() -> None:
    config = LongLiveSystemConfig(
        offload_overlap="d2h_compute",
        onload_overlap="kv_stream",
        gpu_union_cache="per_chunk",
        gpu_union_cache_budget_mib=768,
    )
    assert config.offload_overlap == "d2h_compute"
    assert config.onload_overlap == "kv_stream"


def test_online_context_has_no_teacher_or_full_candidate_fields() -> None:
    names = {item.name for item in fields(OnlineRoutingContext)}
    forbidden = {"query", "key", "value", "dense_output", "dense_attention"}
    assert not names & forbidden
    context = OnlineRoutingContext(
        query_centroids=torch.zeros(1, 2, 3, 4),
        query_group_sizes=torch.ones(1, 2, 3, dtype=torch.long),
        key_prototypes=torch.zeros(1, 2, 5, 4),
        value_prototypes=torch.zeros(1, 2, 5, 4),
        block_frame_ids=torch.arange(5),
        block_token_starts=torch.arange(5) * 4,
        block_token_ends=torch.arange(5) * 4 + 4,
        block_age=torch.arange(5).float(),
    )
    assert context.blocks == 5


def test_offline_teacher_accepts_full_tensors_but_checks_shapes() -> None:
    query = torch.zeros(1, 3, 2, 4)
    key = torch.zeros(1, 5, 2, 4)
    OfflineTeacherContext(query=query, key=key, value=key.clone(), dense_output=query)
    with pytest.raises(ValueError, match="dense_output"):
        OfflineTeacherContext(
            query=query,
            key=key,
            value=key.clone(),
            dense_output=torch.zeros(1, 2, 2, 4),
        )


def test_attention_bias_plan_is_compact_and_normalized() -> None:
    plan = AttentionBiasPlan(
        role_names=("identity", "scene"),
        query_role_probabilities=torch.tensor([[[0.8, 0.2], [0.1, 0.9]]]),
        history_role_probabilities=torch.tensor(
            [[[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]]]
        ),
        history_age_weights=torch.ones(1, 1, 3),
        mode="causal_soft_role",
    )
    assert plan.query_role_probabilities.ndim == 3
    assert plan.digest() == plan.digest()


def test_transfer_plan_preserves_route_and_records_padding() -> None:
    route = route_plan()
    exact = build_transfer_plan(
        route,
        [5, 7],
        frame_tokens=8,
        layout="exact_compact",
        bytes_per_token=16,
    )
    assert exact.route_plan_sha256 == route.digest()
    assert exact.logical_tokens == 3
    assert exact.physical_copy_tokens == 3
    assert exact.padding_tokens == 0
    assert exact.source_run_count == 2
    assert exact.logical_to_physical[0, 0, :3].tolist() == [0, 1, 2]

    blocked = build_transfer_plan(
        route,
        [5, 7],
        frame_tokens=8,
        layout="block64",
        bytes_per_token=16,
    )
    assert blocked.route_plan_sha256 == route.digest()
    assert blocked.physical_copy_tokens == 16
    assert blocked.padding_tokens == 13
    assert blocked.payload_bytes == 48
    assert blocked.physical_copy_bytes == 256


def test_transfer_plan_residency_changes_cost_not_route() -> None:
    route = route_plan()
    resident = torch.tensor([[[True, False, False, False]]])
    plan = build_transfer_plan(
        route,
        [5, 7],
        frame_tokens=8,
        layout="exact_compact",
        bytes_per_token=16,
        resident_logical_mask=resident,
    )
    assert plan.route_plan_sha256 == route.digest()
    assert plan.resident_tokens == 1
    assert plan.missing_logical_tokens == 2


def test_cost_model_is_separate_and_versioned() -> None:
    route = route_plan()
    transfer = build_transfer_plan(
        route,
        [5, 7],
        frame_tokens=8,
        layout="exact_compact",
        bytes_per_token=16,
    )
    profile = HardwareCostProfile(
        profile_id="h200-isolated-v1",
        model_version="piecewise-v1",
        h2d_bytes_per_second=10_000.0,
        hbm_bytes_per_second=100_000.0,
        copy_launch_seconds=1e-4,
        pack_run_seconds=2e-4,
        source_artifact_sha256="a" * 64,
    )
    prediction = SystemCostModel(profile).predict(
        route,
        transfer,
        execution_dataflow="qout_grouped_fa2",
        pipeline_state=CausalPipelineState(predicted_overlap_fraction=0.5),
        query_reuse_factor=2.0,
    )
    assert prediction.profile_id == profile.profile_id
    assert prediction.predicted_exposed_h2d_s == pytest.approx(
        prediction.h2d_service_s * 0.5
    )
    assert mean_absolute_percentage_error([1.0, 2.0], [1.0, 2.5]) == pytest.approx(0.1)


def test_trace_rejects_negative_measured_wait() -> None:
    with pytest.raises(ValueError, match="measured_exposed_wait"):
        SystemTraceRecord(
            layer_id=0,
            chunk_id=0,
            denoising_pass=0,
            route_plan_sha256=None,
            transfer_plan_sha256=None,
            execution_dataflow="qout_grouped_fa2",
            measured_exposed_wait_s=-1.0,
        )
