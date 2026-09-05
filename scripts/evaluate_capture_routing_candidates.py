#!/usr/bin/env python3
"""Evaluate online-legal route candidates against an isolated history-only teacher."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.cost_model import HardwareCostProfile, SystemCostModel
from adapters.longlive_sparse.offline_eval import (
    dense_history_attention,
    output_error_metrics,
    routed_history_attention,
)
from adapters.longlive_sparse.selectors import summarize_query_for_pretransfer
from adapters.longlive_sparse.system_utility_route import (
    SystemUtilityRouteConfig,
    build_cost_model_set_cost_factory,
    build_system_utility_route,
)
from adapters.longlive_sparse.transfer_plan import (
    build_transfer_execution_plan,
    build_transfer_plan,
)
from adapters.longlive_sparse.utility import (
    VALUE_CANDIDATES,
    apply_query_group_policy,
)


def _ordered_frames(frame_ids: torch.Tensor) -> list[int]:
    return list(dict.fromkeys(int(value) for value in frame_ids[0, 0]))


def _method_params(path: Path, method: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload.get("method_params", payload).get(method, {}))


def _profile(path: Path) -> HardwareCostProfile:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("cost_aware_admission_allowed", False):
        raise ValueError("cost-aware admission is disabled by the held-out MAPE gate")
    return HardwareCostProfile(**payload["profile"])


def _route_record(
    route,
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    frame_ids: torch.Tensor,
    token_ids: torch.Tensor,
    teacher: torch.Tensor,
    candidate_frames: list[int],
    frame_tokens: int,
    bytes_per_token: int,
    transfer_layout: str,
    transfer_mode: str,
    model: SystemCostModel | None,
    query_chunk_size: int,
) -> dict:
    started = time.perf_counter()
    candidate_output = routed_history_attention(
        query,
        key,
        value,
        frame_ids,
        token_ids,
        route,
        query_chunk_size=query_chunk_size,
    )
    if query.is_cuda:
        torch.cuda.synchronize(query.device)
    teacher_eval_s = time.perf_counter() - started
    transfer = build_transfer_plan(
        route,
        candidate_frames,
        frame_tokens=frame_tokens,
        layout=transfer_layout,
        bytes_per_token=bytes_per_token,
    )
    execution = build_transfer_execution_plan(transfer, mode=transfer_mode)
    prediction = None
    if model is not None:
        prediction = model.predict(
            route,
            transfer,
            execution_dataflow="qout_grouped_fa2",
            transfer_mode=transfer_mode,
        ).as_dict()
    return {
        "route": route.as_dict(),
        "history_only_output_error": output_error_metrics(teacher, candidate_output),
        "history_only_teacher_eval_s": teacher_eval_s,
        "transfer_plan": transfer.as_dict(),
        "transfer_execution": execution.as_dict(),
        "predicted_cost": prediction,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("query_policy", "utility"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--query-chunk-size", type=int, default=64)
    parser.add_argument("--query-block-size", type=int, default=64)
    parser.add_argument("--exact-k-tokens", type=int, default=9360)
    parser.add_argument("--history-density", type=float, default=0.25)
    parser.add_argument("--source-method", default="transfer_vaware_hybrid_history")
    parser.add_argument("--method-params-file", default="configs/formal/method_params.json")
    parser.add_argument("--cost-profile")
    parser.add_argument("--transfer-layout", default="block64")
    parser.add_argument("--transfer-mode", default="packed_separate")
    parser.add_argument("--group-policy", default="legacy_exact_union")
    parser.add_argument("--group-top-p", type=float, default=0.90)
    parser.add_argument("--group-min-k-ratio", type=float, default=0.10)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but unavailable")
    device = torch.device(args.device)
    capture_path = Path(args.capture)
    capture = torch.load(capture_path, map_location="cpu", weights_only=True)
    query_cpu = capture["query"]
    key_cpu = capture["key"]
    value_cpu = capture["value"]
    frame_ids_cpu = capture["frame_ids"].long()
    token_ids_cpu = capture["token_ids"].long()
    candidate_frames = _ordered_frames(frame_ids_cpu)
    counts = [int((frame_ids_cpu[0, 0] == frame).sum()) for frame in candidate_frames]
    if not counts or len(set(counts)) != 1:
        raise ValueError("capture candidate frames must have uniform token counts")
    frame_tokens = counts[0]
    sparse_config = SparseHistoryConfig(
        method=args.source_method,
        history_density=args.history_density,
        block_size=64,
        refresh_policy="per_chunk",
        rope_policy="upstream_zero",
        method_params=_method_params(Path(args.method_params_file), args.source_method),
    )
    archive = HistoryArchive(sparse_config, spatial_height=30, spatial_width=52)
    for frame in candidate_frames:
        mask = frame_ids_cpu[0, 0] == frame
        order = torch.argsort(token_ids_cpu[0, 0, mask])
        archive.index_frame(
            0,
            frame,
            key_cpu[:, mask][:, order],
            value_cpu[:, mask][:, order],
        )
    summary = summarize_query_for_pretransfer(query_cpu, args.query_block_size)
    source_route = archive.route_indexed(
        0, summary, candidate_frames, exact_k_tokens=args.exact_k_tokens
    )
    model = SystemCostModel(_profile(Path(args.cost_profile))) if args.cost_profile else None
    context = archive.online_routing_context(
        0,
        summary,
        candidate_frames,
        hardware_profile_id=(model.profile.profile_id if model else None),
        cost_model_version=(model.profile.model_version if model else None),
    )
    query = query_cpu.to(device)
    key = key_cpu.to(device)
    value = value_cpu.to(device)
    frame_ids = frame_ids_cpu.to(device)
    token_ids = token_ids_cpu.to(device)
    teacher_started = time.perf_counter()
    teacher = dense_history_attention(
        query,
        key,
        value,
        query_chunk_size=args.query_chunk_size,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dense_teacher_s = time.perf_counter() - teacher_started
    bytes_per_token = 2 * query.shape[-1] * query.element_size()
    records = {}
    if args.mode == "query_policy":
        variants = [
            ("legacy_exact_union", 1.0, 0.0),
            ("top_p_080", 0.80, 0.10),
            ("top_p_090", 0.90, 0.10),
            ("top_p_095", 0.95, 0.10),
        ]
        for name, top_p, minimum in variants:
            route_started = time.perf_counter()
            route = apply_query_group_policy(
                source_route,
                context,
                policy=(
                    "legacy_exact_union"
                    if name == "legacy_exact_union"
                    else "mass_preserving_top_p"
                ),
                top_p=top_p,
                min_k_ratio=minimum,
            )
            route_s = time.perf_counter() - route_started
            records[name] = {
                "online_route_s": route_s,
                **_route_record(
                    route,
                    query=query,
                    key=key,
                    value=value,
                    frame_ids=frame_ids,
                    token_ids=token_ids,
                    teacher=teacher,
                    candidate_frames=candidate_frames,
                    frame_tokens=frame_tokens,
                    bytes_per_token=bytes_per_token,
                    transfer_layout=args.transfer_layout,
                    transfer_mode=args.transfer_mode,
                    model=model,
                    query_chunk_size=args.query_chunk_size,
                ),
            }
    else:
        if model is None:
            raise ValueError("utility mode requires --cost-profile for marginal candidates")
        factory = build_cost_model_set_cost_factory(
            context,
            model,
            exact_k_tokens=args.exact_k_tokens,
            transfer_layout=args.transfer_layout,
            transfer_mode=args.transfer_mode,
            execution_dataflow="qout_grouped_fa2",
        )
        for candidate in sorted(VALUE_CANDIDATES):
            for cost_strategy in ("static_block", "marginal_set"):
                name = f"{candidate}__{cost_strategy}"
                route_started = time.perf_counter()
                route = build_system_utility_route(
                    context,
                    exact_k_tokens=args.exact_k_tokens,
                    config=SystemUtilityRouteConfig(
                        value_candidate=candidate,
                        cost_strategy=cost_strategy,
                        history_density=args.history_density,
                        group_selection_policy=args.group_policy,
                        group_top_p=args.group_top_p,
                        group_min_k_ratio=args.group_min_k_ratio,
                    ),
                    set_cost_factory=(factory if cost_strategy == "marginal_set" else None),
                )
                route_s = time.perf_counter() - route_started
                records[name] = {
                    "online_route_s": route_s,
                    **_route_record(
                        route,
                        query=query,
                        key=key,
                        value=value,
                        frame_ids=frame_ids,
                        token_ids=token_ids,
                        teacher=teacher,
                        candidate_frames=candidate_frames,
                        frame_tokens=frame_tokens,
                        bytes_per_token=bytes_per_token,
                        transfer_layout=args.transfer_layout,
                        transfer_mode=args.transfer_mode,
                        model=model,
                        query_chunk_size=args.query_chunk_size,
                    ),
                }
    payload = {
        "status": "pass",
        "mode": args.mode,
        "capture": str(capture_path.resolve()),
        "capture_metadata": {
            "layer": int(capture.get("layer", -1)),
            "current_start": int(capture.get("current_start", -1)),
            "query_shape": list(query.shape),
            "history_shape": list(key.shape),
            "candidate_frame_ids": candidate_frames,
        },
        "teacher_boundary": "offline history-only full Q/K/V; no exact/current K/V in capture",
        "online_route_inputs": "Q summary plus CPU Block64 K/V prototypes and frozen cost profile only",
        "dense_teacher_s": dense_teacher_s,
        "source_route": source_route.as_dict(),
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "cases": len(records), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
