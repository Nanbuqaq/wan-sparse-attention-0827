#!/usr/bin/env python3
"""Independent correctness gate for the three Stage-3 routes and backends."""

from __future__ import annotations

import argparse
import json
import math
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


def config(method: str, backend: str, density: float) -> MethodConfig:
    return MethodConfig(
        method=method,
        backend=backend,
        density=density,
        q_clusters=8,
        k_clusters=8,
        kmeans_init_iterations=2,
        kmeans_step_iterations=1,
        inference_steps=4,
        calls_per_step=2,
        backend_params={"block_m": 64, "block_n": 32} if backend == "varlen_triton_csr" else {},
        route_params={
            "base_fraction": 0.75,
            "local_fraction": 0.125,
            "remote_clusters": 8,
            "refresh_calls": 2,
            "v_objective": "output_residual",
            "v_weight": 0.75,
            "frames_latent": 4,
            "height_latent": 8,
            "width_latent": 8,
        },
        parameter_origin="stage3_correctness",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="results/metrics/stage3_correctness/correctness.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(3003)
    q = torch.randn(1, 2, 256, 128, device=args.device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    dense = F.scaled_dot_product_attention(q, k, v)
    rows = []
    for method in ("coverage_cluster", "vaware_cluster", "stage3_hybrid"):
        graph_hashes = {}
        for backend in ("fixed64_bf16", "varlen_triton_csr"):
            for density in (1.0, 0.25):
                q_work, k_work, v_work, plan = route_attention(
                    q,
                    k,
                    v,
                    config=config(method, backend, density),
                    state=RoutingState(),
                    layer=0,
                    call_index=0,
                )
                output, kernel_ms, inverse_ms = execute_route(q_work, k_work, v_work, plan)
                error = metrics(dense, output) if density == 1.0 else None
                graph_hashes[(backend, density)] = plan.graph_sha256()
                expected_pairs = int(round(q.shape[0] * q.shape[1] * q.shape[2] * k.shape[2] * density))
                pair_error = abs(plan.logical_pairs - expected_pairs) / (q.shape[0] * q.shape[1] * q.shape[2] * k.shape[2])
                passed = (
                    pair_error <= 1e-4
                    and plan.q_sorted_indices is None
                    and plan.k_sorted_indices is None
                    and (density < 1.0 or (error["relative_l2"] <= 0.01 and error["cosine"] >= 0.9999 and error["allclose_atol_rtol_2e2"]))
                )
                rows.append(
                    {
                        "method": method,
                        "backend": backend,
                        "density": density,
                        "logical_pairs": plan.logical_pairs,
                        "expected_pairs": expected_pairs,
                        "pair_density_error": pair_error,
                        "preserves_original_order": plan.q_sorted_indices is None and plan.k_sorted_indices is None,
                        "route_graph_sha256": plan.graph_sha256(),
                        "attention_vs_dense": error,
                        "kernel_ms": kernel_ms,
                        "inverse_ms": inverse_ms,
                        "status": "pass" if passed else "fail",
                    }
                )
        same_route = graph_hashes[("fixed64_bf16", 0.25)] == graph_hashes[("varlen_triton_csr", 0.25)]
        rows.append({"method": method, "check": "same_route_fixed_vs_csr_d250", "status": "pass" if same_route else "fail", "fixed_hash": graph_hashes[("fixed64_bf16", 0.25)], "csr_hash": graph_hashes[("varlen_triton_csr", 0.25)]})
    payload = {"schema_version": 3, "rows": rows, "status": "pass" if all(row["status"] == "pass" for row in rows) else "fail"}
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "rows": len(rows), "output": str(output)}, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
