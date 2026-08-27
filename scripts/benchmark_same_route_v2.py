#!/usr/bin/env python3
"""Strict same-RoutePlan backend benchmark on captured real Wan activations."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from adapters.kernels import execute_route
from adapters.routing import RoutingState, route_attention
from adapters.types import MethodConfig


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def timed_replay(q, k, v, plan, *, warmup: int, repeats: int) -> dict:
    cold_plan = copy.deepcopy(plan)
    _, cold_kernel, _ = execute_route(q, k, v, cold_plan)
    for _ in range(warmup):
        execute_route(q, k, v, copy.deepcopy(plan))
    kernel_values = []
    inverse_values = []
    planner_values = []
    output = None
    for _ in range(repeats):
        local = copy.deepcopy(plan)
        output, kernel_ms, inverse_ms = execute_route(q, k, v, local)
        kernel_values.append(kernel_ms)
        inverse_values.append(inverse_ms)
        planner_values.append(local.planner_ms)
    return {
        "output": output,
        "cold_kernel_ms": cold_kernel,
        "kernel_p50_ms": statistics.median(kernel_values),
        "kernel_p90_ms": percentile(kernel_values, 0.9),
        "inverse_p50_ms": statistics.median(inverse_values),
        "planner_p50_ms": statistics.median(planner_values),
    }


def error(reference, candidate) -> dict:
    delta = candidate.float() - reference.float()
    return {
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)),
        "cosine": float(torch.nn.functional.cosine_similarity(reference.float().flatten(), candidate.float().flatten(), dim=0)),
    }


def method_config(method: str, backend: str, graph_kind: str) -> MethodConfig:
    if method == "svg2":
        route_params = {"record_route_graph_hash": True}
        q_clusters, k_clusters, init, step = 300, 1000, 50, 2
    elif method == "svoo":
        route_params = {
            "q_clusters": 256,
            "k_clusters": 1024,
            "co_cluster_iterations": 1,
            "reuse_calls": 20,
            "record_route_graph_hash": True,
        }
        q_clusters, k_clusters, init, step = 256, 1024, 1, 1
    else:
        raise ValueError(method)
    if graph_kind == "fixedgraph":
        route_params["materialization"] = "fixed64_graph"
    return MethodConfig(
        method=method,
        backend=backend,
        density=0.25,
        q_clusters=q_clusters,
        k_clusters=k_clusters,
        kmeans_init_iterations=init,
        kmeans_step_iterations=step,
        route_params=route_params,
        backend_params={"block_m": 64, "block_n": 32},
        parameter_origin="frozen_same_route_benchmark_v2",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-manifest", required=True)
    parser.add_argument("--max-points", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    manifest = json.loads(Path(args.capture_manifest).read_text(encoding="utf-8"))
    points = manifest["records"][: args.max_points]
    rows = []
    for point_record in points:
        point = torch.load(point_record["path"], map_location="cpu", weights_only=False)
        q = point["q"].to(args.device)
        k = point["k"].to(args.device)
        v = point["v"].to(args.device)
        for method in ("svg2", "svoo"):
            # Fixed graph: build once in native layout, then replay native and CSR.
            q_fixed, k_fixed, v_fixed, fixed_native_plan = route_attention(
                q, k, v,
                config=method_config(method, "varlen_triton_native", "fixedgraph"),
                state=RoutingState(), layer=int(point["layer"]), call_index=int(point["call_index"]),
            )
            fixed_graph_hash = fixed_native_plan.graph_sha256()
            fixed_results = {}
            for backend in ("varlen_triton_native", "varlen_triton_csr"):
                plan = copy.deepcopy(fixed_native_plan)
                plan.backend = backend
                fixed_results[backend] = timed_replay(q_fixed, k_fixed, v_fixed, plan, warmup=args.warmup, repeats=args.repeats)
            fixed_reference = fixed_results["varlen_triton_native"]["output"]
            for backend, result in fixed_results.items():
                rows.append({
                    "point": point_record["path"], "layer": point["layer"], "call_index": point["call_index"],
                    "method": method, "graph_kind": "fixedgraph", "backend": backend,
                    "route_graph_sha256": fixed_graph_hash,
                    "cold_kernel_ms": result["cold_kernel_ms"], "kernel_p50_ms": result["kernel_p50_ms"],
                    "kernel_p90_ms": result["kernel_p90_ms"], "planner_p50_ms": result["planner_p50_ms"],
                    "inverse_p50_ms": result["inverse_p50_ms"], "output_error_vs_native": error(fixed_reference, result["output"]),
                })

            # True variable graph: construct once and replay native and CSR.
            q_var, k_var, v_var, var_plan = route_attention(
                q, k, v,
                config=method_config(method, "varlen_triton_native", "varlen"),
                state=RoutingState(), layer=int(point["layer"]), call_index=int(point["call_index"]),
            )
            var_hash = var_plan.graph_sha256()
            var_results = {}
            for backend in ("varlen_triton_native", "varlen_triton_csr"):
                plan = copy.deepcopy(var_plan)
                plan.backend = backend
                var_results[backend] = timed_replay(q_var, k_var, v_var, plan, warmup=args.warmup, repeats=args.repeats)
            var_reference = var_results["varlen_triton_native"]["output"]
            for backend, result in var_results.items():
                rows.append({
                    "point": point_record["path"], "layer": point["layer"], "call_index": point["call_index"],
                    "method": method, "graph_kind": "varlen", "backend": backend,
                    "route_graph_sha256": var_hash,
                    "cold_kernel_ms": result["cold_kernel_ms"], "kernel_p50_ms": result["kernel_p50_ms"],
                    "kernel_p90_ms": result["kernel_p90_ms"], "planner_p50_ms": result["planner_p50_ms"],
                    "inverse_p50_ms": result["inverse_p50_ms"], "output_error_vs_native": error(var_reference, result["output"]),
                })
        del q, k, v
        torch.cuda.empty_cache()
    payload = {
        "schema_version": 2,
        "same_route_plan_replayed": True,
        "rows": rows,
        "status": "pass" if rows and all(row["output_error_vs_native"]["relative_l2"] <= 0.01 for row in rows) else "fail",
    }
    output = ROOT / "results/metrics/same_route_kernel_benchmark_v2.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(rows), "output": str(output)}, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

