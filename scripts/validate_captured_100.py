#!/usr/bin/env python3
"""100% backend evidence on captured real Wan activations."""

from __future__ import annotations

import argparse
import json
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


def metrics(reference, candidate):
    first = reference.float()
    second = candidate.float()
    delta = second - first
    return {
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(first).clamp_min(1e-12)),
        "cosine": float(F.cosine_similarity(first.flatten(), second.flatten(), dim=0)),
        "allclose_atol_rtol_2e2": bool(torch.allclose(first, second, atol=2e-2, rtol=2e-2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    point = torch.load(args.capture, map_location="cpu", weights_only=False)
    q = point["q"].to(args.device)
    k = point["k"].to(args.device)
    v = point["v"].to(args.device)
    dense = F.scaled_dot_product_attention(q, k, v)
    cases = [
        ("original_block", "fixed64_bf16", {}, 1, 1),
        ("svg2", "fixed64_bf16", {}, 300, 1000),
        ("svg2", "varlen_triton_native", {}, 300, 1000),
        ("svg2", "varlen_triton_csr", {"block_m": 64, "block_n": 32}, 300, 1000),
    ]
    rows = []
    for method, backend, backend_params, q_clusters, k_clusters in cases:
        q_work, k_work, v_work, plan = route_attention(
            q,
            k,
            v,
            config=MethodConfig(
                method=method,
                backend=backend,
                density=1.0,
                q_clusters=q_clusters,
                k_clusters=k_clusters,
                kmeans_init_iterations=10,
                kmeans_step_iterations=2,
                backend_params=backend_params,
                parameter_origin="captured_100_v2",
            ),
            state=RoutingState(),
            layer=int(point["layer"]),
            call_index=int(point["call_index"]),
        )
        output, kernel_ms, inverse_ms = execute_route(q_work, k_work, v_work, plan)
        error = metrics(dense, output)
        rows.append(
            {
                "method": method,
                "backend": backend,
                "logical_density": plan.logical_density,
                "logical_pairs": plan.logical_pairs,
                "route_graph_sha256": plan.graph_sha256(),
                "attention": error,
                "kernel_ms": kernel_ms,
                "inverse_ms": inverse_ms,
                "status": "pass" if error["relative_l2"] <= 0.01 and error["cosine"] >= 0.9999 and error["allclose_atol_rtol_2e2"] else "fail",
            }
        )
    payload = {
        "schema_version": 2,
        "capture": args.capture,
        "layer": point["layer"],
        "call_index": point["call_index"],
        "head_ids": point["head_ids"],
        "rows": rows,
        "status": "pass" if all(row["status"] == "pass" for row in rows) else "fail",
    }
    output = ROOT / "results/metrics/correctness_v2/captured_100.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": rows, "output": str(output)}, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

