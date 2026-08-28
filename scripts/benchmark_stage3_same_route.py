#!/usr/bin/env python3
"""Replay one Stage-3 RoutePlan across fixed64 and CSR backend parameters."""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from adapters.kernels import execute_route
from adapters.routing import RoutingState, route_attention
from adapters.types import MethodConfig


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = q * (len(values) - 1)
    lower = int(position)
    upper = min(len(values) - 1, lower + 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def error(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    first = reference.float()
    second = candidate.float()
    delta = second - first
    return {
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(first).clamp_min(1e-12)),
        "cosine": float(F.cosine_similarity(first.flatten(), second.flatten(), dim=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", default="results/captures/qkv_v2/chef_motion/seed_000042/layer_09_call_049.pt")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    point = torch.load(ROOT / args.capture, map_location="cpu", weights_only=False)
    q = point["q"].to(args.device)
    k = point["k"].to(args.device)
    v = point["v"].to(args.device)
    config = MethodConfig(
        method="stage3_hybrid",
        backend="fixed64_bf16",
        density=0.25,
        q_clusters=128,
        k_clusters=128,
        kmeans_init_iterations=3,
        kmeans_step_iterations=1,
        inference_steps=50,
        calls_per_step=2,
        route_params={
            "base_fraction": 0.80,
            "local_fraction": 0.10,
            "remote_clusters": 128,
            "refresh_calls": 20,
            "v_objective": "output_residual",
            "v_weight": 0.75,
            "early_base_bonus": 0.05,
            "late_base_bonus": 0.025,
            "frames_latent": 21,
            "height_latent": 30,
            "width_latent": 52,
        },
        parameter_origin="stage3_same_route_benchmark",
    )
    q_fixed, k_fixed, v_fixed, fixed_plan = route_attention(
        q,
        k,
        v,
        config=config,
        state=RoutingState(),
        layer=int(point["layer"]),
        call_index=int(point["call_index"]),
    )
    cases = [("fixed64_bf16", None)] + [("varlen_triton_csr", value) for value in (16, 32, 64, 128)]
    rows = []
    reference = None
    graph_hash = fixed_plan.graph_sha256()
    for backend, block_n in cases:
        metadata = dict(fixed_plan.metadata)
        if backend == "varlen_triton_csr":
            metadata["backend_params"] = {"block_m": 64, "block_n": block_n}
        plan = dataclasses.replace(fixed_plan, backend=backend, metadata=metadata)
        q_work, k_work, v_work = (q_fixed, k_fixed, v_fixed) if backend == "fixed64_bf16" else (q, k, v)
        outputs = []
        times = []
        planners = []
        for index in range(args.warmup + args.repeats):
            output, kernel_ms, _ = execute_route(q_work, k_work, v_work, plan)
            if index >= args.warmup:
                times.append(kernel_ms)
                planners.append(plan.planner_ms)
            outputs.append(output)
        candidate = outputs[-1]
        if reference is None:
            reference = candidate
        rows.append(
            {
                "backend": backend,
                "block_m": 64,
                "block_n": block_n,
                "route_graph_sha256": plan.graph_sha256(),
                "same_route_graph": plan.graph_sha256() == graph_hash,
                "kernel_p50_ms": statistics.median(times),
                "kernel_p90_ms": percentile(times, 0.9),
                "planner_p50_ms": statistics.median(planners),
                "combined_p50_ms": statistics.median([a + b for a, b in zip(times, planners)]),
                "output_vs_fixed": error(reference, candidate),
                "logical_density": plan.logical_density,
                "scheduled_density": plan.scheduled_density_vs_dense,
            }
        )
        del outputs, candidate
        torch.cuda.empty_cache()
    best = min(rows, key=lambda row: row["combined_p50_ms"])
    payload = {
        "schema_version": 3,
        "capture": args.capture,
        "route_graph_sha256": graph_hash,
        "rows": rows,
        "selected_backend": {"backend": best["backend"], "block_m": best["block_m"], "block_n": best["block_n"]},
        "status": "pass" if all(row["same_route_graph"] and row["output_vs_fixed"]["relative_l2"] <= 0.01 for row in rows) else "fail",
    }
    output = ROOT / "results/metrics/stage3_same_route_kernel.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
