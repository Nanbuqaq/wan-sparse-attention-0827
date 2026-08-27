#!/usr/bin/env python3
"""Screen method parameter candidates on captured real Wan Q/K/V."""

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


def candidates() -> list[dict]:
    rows = []

    def add(method, candidate, route_params, *, q=128, k=128, init=3, step=1):
        rows.append(
            {
                "method": method,
                "candidate": candidate,
                "route_params": route_params,
                "q_clusters": q,
                "k_clusters": k,
                "init_iterations": init,
                "step_iterations": step,
            }
        )

    add("svg2", "q100_k400_i10", {}, q=100, k=400, init=10, step=1)
    add("svg2", "q200_k800_i25", {}, q=200, k=800, init=25, step=2)
    add("svg2", "q300_k1000_i50", {}, q=300, k=1000, init=50, step=2)
    add("adacluster", "q64_k64_t05", {"q_clusters": 64, "initial_k_clusters": 64, "max_added_clusters": 64, "distance_threshold": 0.5, "reuse_calls": 20}, q=64, k=128, init=3)
    add("adacluster", "q100_k100_t10", {"q_clusters": 100, "initial_k_clusters": 100, "max_added_clusters": 64, "distance_threshold": 1.0, "reuse_calls": 20}, q=100, k=164, init=5)
    add("adacluster", "q100_k100_t55", {"q_clusters": 100, "initial_k_clusters": 100, "max_added_clusters": 64, "distance_threshold": 5.5, "reuse_calls": 20}, q=100, k=164, init=5)
    add("svoo", "q128_k512_c1", {"q_clusters": 128, "k_clusters": 512, "co_cluster_iterations": 1, "reuse_calls": 10}, q=128, k=512, init=1)
    add("svoo", "q256_k1024_c1", {"q_clusters": 256, "k_clusters": 1024, "co_cluster_iterations": 1, "reuse_calls": 20}, q=256, k=1024, init=1)
    add("svoo", "q256_k1024_c2", {"q_clusters": 256, "k_clusters": 1024, "co_cluster_iterations": 2, "reuse_calls": 20}, q=256, k=1024, init=2)
    add("scope", "q64_s128_i3", {"q_clusters": 64, "subspace_clusters": 128}, q=64, k=128, init=3)
    add("scope", "q100_s256_i3", {"q_clusters": 100, "subspace_clusters": 256}, q=100, k=256, init=3)
    add("scope", "q100_s333_i5", {"q_clusters": 100, "subspace_clusters": 333}, q=100, k=333, init=5)
    for factor in (1.25, 1.5, 2.0):
        add("capacity_balanced", f"k128_c{factor}", {"clusters": 128, "capacity_factor": factor})
    for threshold in (2.5, 4.0, 5.5):
        add("radius_adaptive", f"k64_a64_t{threshold}", {"base_clusters": 64, "max_added_clusters": 64, "radius_threshold": threshold, "reuse_calls": 20})
    for coarse, branches in ((16, 4), (32, 4), (64, 2)):
        add("hierarchical", f"c{coarse}_b{branches}", {"coarse_clusters": coarse, "branches": branches})
    for subspaces, codebook in ((4, 8), (4, 16), (8, 8)):
        add("product_quantized", f"m{subspaces}_k{codebook}", {"subspaces": subspaces, "codebook_clusters": codebook})
    for weight in (0.1, 0.25, 0.5):
        add("spatiotemporal", f"k128_w{weight}", {"clusters": 128, "position_weight": weight, "frames_latent": 21, "height_latent": 30, "width_latent": 52})
    for rank in (16, 32, 64):
        add("query_metric", f"k128_r{rank}", {"clusters": 128, "rank": rank, "basis_refresh_calls": 20})
    return rows


def attention_mass_recall(q, k, plan, samples=32) -> float:
    length = int(plan.metadata["original_length"])
    q = q[:, :, :length]
    k = k[:, :, :length]
    ids = torch.linspace(0, length - 1, min(samples, length), device=q.device).round().long()
    recalls = []
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            q_edges = plan.q_sizes[b, h].cumsum(0).long()
            k_edges = torch.cat((torch.zeros(1, device=q.device, dtype=torch.long), plan.k_sizes[b, h].cumsum(0).long()))
            q_blocks = torch.bucketize(ids, q_edges, right=False).clamp_max(plan.q_sizes.shape[-1] - 1)
            scores = q[b, h].index_select(0, ids).float() @ k[b, h].float().T / math.sqrt(q.shape[-1])
            probabilities = torch.softmax(scores, dim=-1)
            for row, q_block in enumerate(q_blocks.tolist()):
                selected = torch.zeros(length, device=q.device, dtype=torch.bool)
                for k_block in plan.block_map[b, h, q_block].nonzero(as_tuple=False).flatten().tolist():
                    selected[k_edges[k_block] : k_edges[k_block + 1]] = True
                recalls.append(float(probabilities[row, selected].sum()))
    return sum(recalls) / len(recalls)


def tensor_error(reference, candidate):
    first = reference.float()
    second = candidate.float()
    delta = second - first
    return {
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(first).clamp_min(1e-12)),
        "cosine": float(F.cosine_similarity(first.flatten(), second.flatten(), dim=0)),
    }


def select_two(rows: list[dict]) -> list[str]:
    summaries = {}
    for row in rows:
        key = (row["method"], row["candidate"])
        value = summaries.setdefault(key, {"errors": [], "recalls": [], "route_ms": [], "e2e_ms": []})
        value["errors"].append(row["output_error"]["relative_l2"])
        value["recalls"].append(row["attention_mass_recall"])
        value["route_ms"].append(row["route_ms"])
        value["e2e_ms"].append(row["route_ms"] + row["kernel_ms"] + row["inverse_ms"])
    selected = []
    methods = sorted({method for method, _ in summaries})
    for method in methods:
        candidates_for_method = []
        for (candidate_method, candidate), values in summaries.items():
            if candidate_method != method:
                continue
            candidates_for_method.append(
                {
                    "candidate": candidate,
                    "error": sum(values["errors"]) / len(values["errors"]),
                    "recall": sum(values["recalls"]) / len(values["recalls"]),
                    "route_ms": sum(values["route_ms"]) / len(values["route_ms"]),
                    "e2e_ms": sum(values["e2e_ms"]) / len(values["e2e_ms"]),
                }
            )
        best_error = min(candidates_for_method, key=lambda item: (item["error"], -item["recall"], item["e2e_ms"]))
        best_speed = min(candidates_for_method, key=lambda item: (item["e2e_ms"], item["error"]))
        chosen = [best_error["candidate"]]
        if best_speed["candidate"] not in chosen:
            chosen.append(best_speed["candidate"])
        if len(chosen) < 2:
            remaining = sorted(candidates_for_method, key=lambda item: (item["error"], item["e2e_ms"]))
            chosen.append(next(item["candidate"] for item in remaining if item["candidate"] not in chosen))
        selected.extend(f"{method}:{candidate}" for candidate in chosen[:2])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-manifest", required=True)
    parser.add_argument("--max-points", type=int, default=2)
    parser.add_argument("--densities", default="0.10,0.25")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    manifest = json.loads(Path(args.capture_manifest).read_text(encoding="utf-8"))
    records = manifest["records"][: args.max_points]
    densities = [float(value) for value in args.densities.split(",")]
    output_rows = []
    for record in records:
        point = torch.load(record["path"], map_location="cpu", weights_only=False)
        q = point["q"].to(args.device)
        k = point["k"].to(args.device)
        v = point["v"].to(args.device)
        dense = F.scaled_dot_product_attention(q, k, v)
        for candidate in candidates():
            for density in densities:
                config = MethodConfig(
                    method=candidate["method"],
                    backend="fixed64_bf16",
                    density=density,
                    q_clusters=candidate["q_clusters"],
                    k_clusters=candidate["k_clusters"],
                    kmeans_init_iterations=candidate["init_iterations"],
                    kmeans_step_iterations=candidate["step_iterations"],
                    route_params=candidate["route_params"],
                    parameter_origin="captured_qkv_candidate",
                )
                try:
                    q_work, k_work, v_work, plan = route_attention(
                        q,
                        k,
                        v,
                        config=config,
                        state=RoutingState(),
                        layer=int(record["layer"]),
                        call_index=int(record["call_index"]),
                    )
                    sparse, kernel_ms, inverse_ms = execute_route(q_work, k_work, v_work, plan)
                    output_rows.append(
                        {
                            "status": "completed",
                            "point": record["path"],
                            "layer": record["layer"],
                            "call_index": record["call_index"],
                            "density": density,
                            **candidate,
                            "actual_density": plan.logical_density,
                            "scheduled_density": plan.scheduled_density_vs_dense,
                            "padding_ratio": plan.padding_ratio,
                            "cluster_size_min": int(plan.k_sizes.min()),
                            "cluster_size_max": int(plan.k_sizes.max()),
                            "route_ms": plan.cluster_ms + plan.permutation_ms + plan.selection_ms,
                            "kernel_ms": kernel_ms,
                            "inverse_ms": inverse_ms,
                            "attention_mass_recall": attention_mass_recall(q_work, k_work, plan),
                            "output_error": tensor_error(dense, sparse),
                            "metadata": plan.metadata,
                        }
                    )
                except Exception as error:
                    output_rows.append(
                        {
                            "status": "failed",
                            "point": record["path"],
                            "layer": record["layer"],
                            "call_index": record["call_index"],
                            "density": density,
                            **candidate,
                            "error": repr(error),
                        }
                    )
        del q, k, v, dense
        torch.cuda.empty_cache()
    completed = [row for row in output_rows if row["status"] == "completed"]
    selected = select_two(completed)
    payload = {
        "schema_version": 2,
        "capture_manifest": args.capture_manifest,
        "points": len(records),
        "densities": densities,
        "rows": output_rows,
        "selected_two_per_method": selected,
        "failed_rows": sum(row["status"] == "failed" for row in output_rows),
        "status": "pass" if len(selected) == 20 else "fail",
    }
    output = ROOT / "results" / "metrics" / "captured_qkv_screen_v2.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(output_rows), "failed_rows": payload["failed_rows"], "selected": selected, "output": str(output)}, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
