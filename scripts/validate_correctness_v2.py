#!/usr/bin/env python3
"""Route-level and backend-shape correctness evidence for suite-v2."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from adapters.kernels import execute_route
from adapters.routing import RoutingState, _plan_metrics, inverse_permute, route_attention
from adapters.types import MethodConfig


def error_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    first = reference.float()
    second = candidate.float()
    delta = second - first
    relative_l2 = torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(first).clamp_min(1e-12)
    cosine = F.cosine_similarity(first.flatten(), second.flatten(), dim=0)
    return {
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "relative_l2": float(relative_l2),
        "cosine": float(cosine),
        "allclose_atol_rtol_2e2": bool(torch.allclose(first, second, atol=2e-2, rtol=2e-2)),
        "finite": bool(torch.isfinite(second).all()),
    }


def element_mask(plan) -> torch.Tensor:
    batch, heads, _, _ = plan.block_map.shape
    length = int(plan.q_sizes[0, 0].sum())
    output = torch.zeros((batch, heads, length, length), device=plan.block_map.device, dtype=torch.bool)
    for b in range(batch):
        for h in range(heads):
            q_edges = torch.cat((torch.zeros(1, device=output.device, dtype=torch.long), plan.q_sizes[b, h].cumsum(0).long()))
            k_edges = torch.cat((torch.zeros(1, device=output.device, dtype=torch.long), plan.k_sizes[b, h].cumsum(0).long()))
            active = plan.block_map[b, h].nonzero(as_tuple=False)
            for qi, ki in active.tolist():
                output[b, h, q_edges[qi] : q_edges[qi + 1], k_edges[ki] : k_edges[ki + 1]] = True
    return output


def masked_reference(q, k, v, plan):
    length = int(plan.metadata["original_length"])
    q = q[:, :, :length]
    k = k[:, :, :length]
    v = v[:, :, :length]
    mask = element_mask(plan)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    output = torch.matmul(torch.softmax(scores.masked_fill(~mask, -float("inf")), dim=-1), v.float()).to(q.dtype)
    if plan.q_sorted_indices is not None:
        output = inverse_permute(output, plan.q_sorted_indices)
    return output


ROUTES = [
    ("original_block", {}),
    ("random_block", {}),
    ("local_3d", {"frames_latent": 4, "height_latent": 8, "width_latent": 8}),
    ("fixed_k128", {}),
    ("fixed_k256", {}),
    ("qsort_local8", {}),
    ("token_oracle", {}),
    ("svg2", {}),
    ("adacluster", {"q_clusters": 4, "initial_k_clusters": 4, "max_added_clusters": 4, "distance_threshold": 4.0}),
    ("svoo", {"q_clusters": 4, "k_clusters": 8, "co_cluster_iterations": 1}),
    ("scope", {"q_clusters": 4, "subspace_clusters": 4}),
    ("capacity_balanced", {"clusters": 8, "capacity_factor": 1.5}),
    ("radius_adaptive", {"base_clusters": 4, "max_added_clusters": 4, "radius_threshold": 4.0}),
    ("hierarchical", {"coarse_clusters": 2, "branches": 2}),
    ("product_quantized", {"subspaces": 4, "codebook_clusters": 2}),
    ("spatiotemporal", {"clusters": 8, "frames_latent": 4, "height_latent": 8, "width_latent": 8}),
    ("query_metric", {"clusters": 8, "rank": 16, "basis_refresh_calls": 2}),
]


def route_checks(device: str) -> list[dict]:
    rows = []
    for index, (method, route_params) in enumerate(ROUTES):
        torch.manual_seed(100 + index)
        q = torch.randn(1, 1, 256, 128, device=device, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        clusters = 256 if method == "fixed_k256" else 8
        config = MethodConfig(
            method=method,
            backend="fixed64_bf16",
            density=1.0,
            q_clusters=4,
            k_clusters=clusters,
            kmeans_init_iterations=2,
            kmeans_step_iterations=1,
            route_params=route_params,
        )
        started = time.perf_counter()
        q_work, k_work, v_work, plan = route_attention(
            q, k, v, config=config, state=RoutingState(), layer=0, call_index=0
        )
        output, kernel_ms, inverse_ms = execute_route(q_work, k_work, v_work, plan)
        reference = F.scaled_dot_product_attention(q, k, v)
        roundtrip = {}
        if plan.q_sorted_indices is not None:
            restored_q = inverse_permute(q_work[:, :, : q.shape[2]], plan.q_sorted_indices)
            roundtrip["q_max_abs"] = float((restored_q - q).abs().max())
        else:
            roundtrip["q_max_abs"] = 0.0
        if plan.k_sorted_indices is not None:
            restored_k = inverse_permute(k_work[:, :, : k.shape[2]], plan.k_sorted_indices)
            restored_v = inverse_permute(v_work[:, :, : v.shape[2]], plan.k_sorted_indices)
            roundtrip["k_max_abs"] = float((restored_k - k).abs().max())
            roundtrip["v_max_abs"] = float((restored_v - v).abs().max())
        else:
            roundtrip.update({"k_max_abs": 0.0, "v_max_abs": 0.0})
        expected_pairs = q.shape[0] * q.shape[1] * q.shape[2] * k.shape[2]
        errors = error_metrics(reference, output)
        passed = (
            plan.logical_pairs == expected_pairs
            and plan.logical_density == 1.0
            and all(value == 0.0 for value in roundtrip.values())
            and errors["allclose_atol_rtol_2e2"]
            and errors["relative_l2"] <= 0.01
            and errors["cosine"] >= 0.9999
        )
        rows.append(
            {
                "method": method,
                "backend": plan.backend,
                "route_graph_sha256": plan.graph_sha256(),
                "logical_pairs": plan.logical_pairs,
                "expected_pairs": expected_pairs,
                "roundtrip": roundtrip,
                "attention": errors,
                "kernel_ms": kernel_ms,
                "inverse_ms": inverse_ms,
                "elapsed_s": time.perf_counter() - started,
                "status": "pass" if passed else "fail",
            }
        )
    return rows


def balanced_sizes(count: int, total: int, device: str) -> torch.Tensor:
    base, extra = divmod(total, count)
    values = [base + int(index < extra) for index in range(count)]
    return torch.tensor(values, dtype=torch.int32, device=device)


def shape_sizes(count: int, total: int, pattern: str, device: str) -> torch.Tensor:
    if pattern == "balanced":
        return balanced_sizes(count, total, device)
    values = torch.zeros(count, dtype=torch.int32, device=device)
    if pattern == "tail":
        values[:] = total // count
        values[-1] += total - int(values.sum())
        return values
    if pattern == "imbalanced_zero":
        seeds = [0, 1, 3, 17, 31, 64, 65, 127, 129, 257, 512]
    elif pattern == "super_cluster":
        seeds = [4096, 2048, 512, 257, 129, 65, 31, 17, 3, 1, 0]
    else:
        raise ValueError(pattern)
    for index, value in enumerate(seeds[:count]):
        values[index] = value
    remaining = total - int(values.sum())
    if remaining < 0:
        raise ValueError("shape seeds exceed total")
    active = max(1, count - len(seeds))
    base, extra = divmod(remaining, active)
    for offset in range(active):
        values[len(seeds) + offset] = base + int(offset < extra)
    return values


BACKEND_CASES = [
    ("small_tail_257", 4, 8, 257, "tail"),
    ("tail_1025", 64, 128, 1025, "tail"),
    ("balanced_wan", 128, 512, 32760, "balanced"),
    ("imbalanced_zero_wan", 256, 1024, 32760, "imbalanced_zero"),
    ("svg2_count_wan", 300, 1000, 32760, "balanced"),
    ("super_cluster_wan", 64, 128, 32760, "super_cluster"),
]


def backend_checks(device: str, include_full: bool) -> list[dict]:
    rows = []
    cases = BACKEND_CASES if include_full else BACKEND_CASES[:2]
    for case_name, q_count, k_count, length, pattern in cases:
        for backend in ("varlen_triton_native", "varlen_triton_csr"):
            torch.manual_seed(q_count + k_count + length)
            q = torch.randn(1, 1, length, 128, device=device, dtype=torch.bfloat16)
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            q_sizes = shape_sizes(q_count, length, pattern, device).view(1, 1, q_count)
            k_sizes = shape_sizes(k_count, length, pattern, device).view(1, 1, k_count)
            block_map = torch.ones((1, 1, q_count, k_count), device=device, dtype=torch.bool)
            plan = _plan_metrics(
                method="backend_shape_validation",
                backend=backend,
                parameter_origin="correctness_v2",
                density=1.0,
                block_map=block_map,
                q_sizes=q_sizes,
                k_sizes=k_sizes,
                q_sorted_indices=None,
                k_sorted_indices=None,
                cluster_ms=0.0,
                permutation_ms=0.0,
                selection_ms=0.0,
                metadata={"original_length": length, "backend_params": {"block_m": 64, "block_n": 32}},
            )
            started = time.perf_counter()
            output, kernel_ms, _ = execute_route(q, k, v, plan)
            reference = F.scaled_dot_product_attention(q, k, v)
            errors = error_metrics(reference, output)
            passed = (
                plan.logical_density == 1.0
                and errors["allclose_atol_rtol_2e2"]
                and errors["relative_l2"] <= 0.01
                and errors["cosine"] >= 0.9999
            )
            rows.append(
                {
                    "case": case_name,
                    "backend": backend,
                    "sequence_length": length,
                    "q_cluster_count": q_count,
                    "k_cluster_count": k_count,
                    "q_size_min": int(q_sizes.min()),
                    "q_size_max": int(q_sizes.max()),
                    "k_size_min": int(k_sizes.min()),
                    "k_size_max": int(k_sizes.max()),
                    "zero_q_clusters": int((q_sizes == 0).sum()),
                    "zero_k_clusters": int((k_sizes == 0).sum()),
                    "route_graph_sha256": plan.graph_sha256(),
                    "attention": errors,
                    "kernel_ms": kernel_ms,
                    "elapsed_s": time.perf_counter() - started,
                    "status": "pass" if passed else "fail",
                }
            )
            del q, k, v, output, reference
            torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--include-full-shapes", action="store_true")
    parser.add_argument("--output", default="results/metrics/correctness_v2/correctness.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    payload = {
        "schema_version": 2,
        "route_checks": route_checks(args.device),
        "backend_checks": backend_checks(args.device, args.include_full_shapes),
    }
    all_rows = payload["route_checks"] + payload["backend_checks"]
    payload["status"] = "pass" if all(row["status"] == "pass" for row in all_rows) else "fail"
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    route_dir = output.parent / "routes"
    backend_dir = output.parent / "backends"
    route_dir.mkdir(parents=True, exist_ok=True)
    backend_dir.mkdir(parents=True, exist_ok=True)
    for row in payload["route_checks"]:
        (route_dir / f"{row['method']}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    for row in payload["backend_checks"]:
        (backend_dir / f"{row['case']}__{row['backend']}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({"status": payload["status"], "route_checks": len(payload["route_checks"]), "backend_checks": len(payload["backend_checks"]), "output": str(output)}, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
