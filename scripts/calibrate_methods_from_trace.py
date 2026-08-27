#!/usr/bin/env python3
"""Calibrate LongLive paper-method parameters on captured Q/K, never formal prompts."""

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

from adapters.longlive_sparse.ar_routing import route_history


GRIDS = {
    "svg2_ar": [
        {"q_clusters": q, "k_clusters": k, "iterations": i}
        for q in (100, 300)
        for k in (200, 512, 1000)
        for i in (2, 5)
    ],
    "adacluster_ar": [
        {
            "q_clusters": q,
            "k_clusters": k,
            "threshold": kt,
            "query_threshold": qt,
        }
        for q in (65, 100)
        for k in (100, 256)
        for kt in (5.5, 7.0)
        for qt in (9.0, 11.0)
    ],
    "svoo_ar": [
        {
            "q_clusters": q,
            "k_clusters": k,
            "iterations": 2,
            "co_cluster_iterations": i,
        }
        for q in (64, 256)
        for k in (256, 512, 1024)
        for i in (1, 2)
    ],
    "scope_ar": [
        {"q_clusters": q, "k_clusters": k, "top_p": p}
        for q in (64, 100)
        for k in (128, 256, 333)
        for p in (0.85, 0.90)
    ],
}


def _selected_candidate_indices(
    plan,
    frame_ids: torch.Tensor,
    token_ids: torch.Tensor,
    batch_index: int,
    head: int,
    group: int,
) -> torch.Tensor:
    count = int(plan.group_history_counts[batch_index, head, group])
    union_slots = plan.group_union_indices[batch_index, head, group, :count]
    union_frames = plan.union_frame_ids[batch_index, head].index_select(0, union_slots)
    union_tokens = plan.union_token_ids[batch_index, head].index_select(0, union_slots)
    max_token = max(int(token_ids.max()), int(union_tokens.max())) + 1
    candidate_codes = frame_ids[batch_index, head].long() * max_token + token_ids[
        batch_index, head
    ].long()
    union_codes = union_frames.long() * max_token + union_tokens.long()
    indices = torch.searchsorted(candidate_codes.contiguous(), union_codes.contiguous())
    if not torch.equal(candidate_codes.index_select(0, indices), union_codes):
        raise KeyError("route plan contains a coordinate outside the capture")
    return indices


def token_recall(
    plan,
    query: torch.Tensor,
    key: torch.Tensor,
    frame_ids: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    density: float,
    max_queries: int,
) -> float:
    budget = max(1, min(key.shape[1], round(key.shape[1] * density)))
    recalls = []
    for batch_index in range(query.shape[0]):
        for head in range(query.shape[2]):
            query_count = min(query.shape[1], max_queries)
            query_indices = torch.linspace(
                0, query.shape[1] - 1, query_count, dtype=torch.long
            ).unique()
            q = query[batch_index, query_indices, head].float()
            k = key[batch_index, :, head].float()
            exact = torch.topk(q @ k.T, k=budget, dim=1).indices
            labels = plan.query_labels[batch_index, head].index_select(0, query_indices)
            for row, group in enumerate(labels.tolist()):
                selected = _selected_candidate_indices(
                    plan,
                    frame_ids,
                    token_ids,
                    batch_index,
                    head,
                    int(group),
                )
                overlap = torch.isin(exact[row], selected).sum()
                recalls.append(float(overlap) / budget)
    return sum(recalls) / len(recalls) if recalls else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--freeze-output")
    parser.add_argument("--density", type=float, default=0.25)
    parser.add_argument("--head-limit", type=int, default=2)
    parser.add_argument("--recall-queries", type=int, default=16)
    parser.add_argument("--methods", default=",".join(GRIDS))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    requested_methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    unknown = set(requested_methods) - set(GRIDS)
    if unknown:
        raise ValueError(f"unknown calibration methods: {sorted(unknown)}")

    records = []
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA calibration requested but CUDA is unavailable")
    device = torch.device(device_name)
    for capture_path in args.capture:
        trace = torch.load(capture_path, map_location="cpu", weights_only=True)
        heads = min(int(args.head_limit), trace["query"].shape[2])
        query = trace["query"][:, :, :heads].float().to(device)
        key = trace["key"][:, :, :heads].float().to(device)
        frame_ids = trace["frame_ids"][:, :heads].to(device)
        token_ids = trace["token_ids"][:, :heads].to(device)
        for method in requested_methods:
            for parameters in GRIDS[method]:
                start = time.perf_counter()
                try:
                    plan = route_history(
                        query,
                        key,
                        frame_ids,
                        token_ids,
                        method=method,
                        density=args.density,
                        exact_k_tokens=query.shape[1],
                        seed=20260827,
                        spec_override=parameters,
                    )
                    recall = token_recall(
                        plan,
                        query,
                        key,
                        frame_ids,
                        token_ids,
                        density=args.density,
                        max_queries=args.recall_queries,
                    )
                    status, error = "pass", None
                except Exception as exception:
                    plan, recall, status = None, None, "fail"
                    error = f"{type(exception).__name__}: {exception}"
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                records.append(
                    {
                        "capture": str(Path(capture_path).resolve()),
                        "method": method,
                        "parameters": parameters,
                        "status": status,
                        "error": error,
                        "elapsed_s": time.perf_counter() - start,
                        "token_recall": recall,
                        "route": plan.as_dict() if plan else None,
                    }
                )

    recommendations = {}
    for method in requested_methods:
        candidates = [
            record
            for record in records
            if record["method"] == method and record["status"] == "pass"
        ]
        if not candidates:
            recommendations[method] = {"status": "fail", "parameters": None}
            continue
        best = min(
            candidates,
            key=lambda item: (
                -float(item["token_recall"]),
                abs(float(item["route"]["history_pair_density"]) - args.density),
                float(item["elapsed_s"]),
                float(item["route"]["history_transfer_density"]),
            ),
        )
        recommendations[method] = {
            "status": "selected_for_smoke",
            "parameters": best["parameters"],
            "token_recall": best["token_recall"],
            "elapsed_s": best["elapsed_s"],
            "route": best["route"],
        }

    payload = {
        "status": "calibration_only",
        "formal_prompts_used": False,
        "selection_rule": "max token recall, then density error, route time, transfer density",
        "head_limit": args.head_limit,
        "device": str(device),
        "recall_queries": args.recall_queries,
        "records": records,
        "recommendations": recommendations,
    }
    Path(args.output).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.freeze_output:
        frozen = {
            "status": "frozen_before_method_smoke",
            "source": str(Path(args.output).resolve()),
            "formal_prompts_used": False,
            "method_params": {
                method: value["parameters"]
                for method, value in recommendations.items()
                if value["parameters"] is not None
            },
        }
        Path(args.freeze_output).write_text(
            json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "records": len(records),
                "output": args.output,
                "freeze_output": args.freeze_output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
