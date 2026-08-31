#!/usr/bin/env python3
"""Calibrate proposed pre-transfer routes with exact offline output teachers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.selectors import (
    build_frame_index,
    route_indexed_history,
    summarize_query_for_pretransfer,
)


SPLITS = ((0.70, 0.15), (0.80, 0.10))
CLUSTERS = (64, 128, 256)
V_WEIGHTS = (0.50, 0.75, 1.00)
TRANSFER_MULTIPLIERS = (1.00, 1.25, 1.50)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reconstruct_frames(
    payload: dict, *, spatial_height: int, spatial_width: int
) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    key = payload["key"].detach().to("cpu")
    value = payload["value"].detach().to("cpu")
    frame_ids = payload["frame_ids"].detach().to("cpu").long()
    token_ids = payload["token_ids"].detach().to("cpu").long()
    if key.shape != value.shape or key.ndim != 4:
        raise ValueError("capture key/value must share [B,K,H,D]")
    batch, _, heads, dim = key.shape
    frame_tokens = spatial_height * spatial_width
    output = []
    for frame_id in sorted({int(value) for value in frame_ids.reshape(-1).tolist()}):
        frame_key = torch.empty((batch, frame_tokens, heads, dim), dtype=key.dtype)
        frame_value = torch.empty_like(frame_key)
        filled = torch.zeros((batch, heads, frame_tokens), dtype=torch.bool)
        for batch_index in range(batch):
            for head in range(heads):
                source = torch.nonzero(
                    frame_ids[batch_index, head] == frame_id, as_tuple=False
                ).flatten()
                target = token_ids[batch_index, head].index_select(0, source)
                if target.numel() != frame_tokens or set(target.tolist()) != set(
                    range(frame_tokens)
                ):
                    raise ValueError(
                        f"frame {frame_id} head {head} is not a complete {frame_tokens}-token frame"
                    )
                frame_key[batch_index, target, head] = key[
                    batch_index, source, head
                ]
                frame_value[batch_index, target, head] = value[
                    batch_index, source, head
                ]
                filled[batch_index, head, target] = True
        if not bool(filled.all()):
            raise RuntimeError(f"frame {frame_id} reconstruction is incomplete")
        output.append((frame_id, frame_key, frame_value))
    return output


def build_indices(
    frames: list[tuple[int, torch.Tensor, torch.Tensor]],
    *,
    clusters: int,
    iterations: int,
    spatial_height: int,
    spatial_width: int,
) -> list:
    config = SparseHistoryConfig(
        method="coverage_cluster_history",
        history_density=0.25,
        block_size=64,
        method_params={"remote_clusters": clusters, "iterations": iterations},
    )
    return [
        build_frame_index(
            frame_id,
            key,
            value,
            key,
            config,
            spatial_height=spatial_height,
            spatial_width=spatial_width,
        )
        for frame_id, key, value in frames
    ]


def candidate_index_lookup(payload: dict) -> list[list[dict[tuple[int, int], int]]]:
    frame_ids = payload["frame_ids"].detach().to("cpu").long()
    token_ids = payload["token_ids"].detach().to("cpu").long()
    batch, heads, tokens = frame_ids.shape
    output: list[list[dict[tuple[int, int], int]]] = []
    for batch_index in range(batch):
        per_head = []
        for head in range(heads):
            mapping = {
                (int(frame_ids[batch_index, head, token]), int(token_ids[batch_index, head, token])): token
                for token in range(tokens)
            }
            if len(mapping) != tokens:
                raise ValueError("capture candidate coordinates are not unique")
            per_head.append(mapping)
        output.append(per_head)
    return output


def teacher_metrics(
    payload: dict,
    plan,
    *,
    sample_queries: int,
) -> dict:
    query = payload["query"].detach().to("cpu").float()
    key = payload["key"].detach().to("cpu").float()
    value = payload["value"].detach().to("cpu").float()
    lookup = candidate_index_lookup(payload)
    batch, query_tokens, heads, dim = query.shape
    sample_ids = torch.linspace(0, query_tokens - 1, sample_queries).round().long()
    relative_l2 = []
    cosines = []
    mass_recalls = []
    for batch_index in range(batch):
        for head in range(heads):
            dense_key = key[batch_index, :, head]
            dense_value = value[batch_index, :, head]
            for query_id in sample_ids.tolist():
                q = query[batch_index, query_id, head]
                dense_logits = q @ dense_key.T / math.sqrt(dim)
                dense_probability = torch.softmax(dense_logits, dim=0)
                dense_output = dense_probability @ dense_value
                group = int(plan.query_labels[batch_index, head, query_id])
                count = int(plan.group_history_counts[batch_index, head, group])
                union_indices = plan.group_union_indices[
                    batch_index, head, group, :count
                ]
                candidate_indices = []
                for union_index in union_indices.tolist():
                    coordinate = (
                        int(plan.union_frame_ids[batch_index, head, union_index]),
                        int(plan.union_token_ids[batch_index, head, union_index]),
                    )
                    candidate_indices.append(lookup[batch_index][head][coordinate])
                selected = torch.tensor(candidate_indices, dtype=torch.long)
                sparse_logits = dense_logits.index_select(0, selected)
                sparse_output = torch.softmax(sparse_logits, dim=0) @ dense_value.index_select(
                    0, selected
                )
                relative_l2.append(
                    float(
                        torch.linalg.vector_norm(sparse_output - dense_output)
                        / torch.linalg.vector_norm(dense_output).clamp_min(1e-8)
                    )
                )
                cosines.append(
                    float(
                        F.cosine_similarity(
                            sparse_output.view(1, -1), dense_output.view(1, -1)
                        )
                    )
                )
                mass_recalls.append(float(dense_probability.index_select(0, selected).sum()))
    values = torch.tensor(relative_l2)
    return {
        "teacher_relative_l2_mean": float(values.mean()),
        "teacher_relative_l2_p90": float(torch.quantile(values, 0.90)),
        "teacher_relative_l2_max": float(values.max()),
        "teacher_cosine_mean": float(torch.tensor(cosines).mean()),
        "attention_mass_recall_mean": float(torch.tensor(mass_recalls).mean()),
        "teacher_queries": len(relative_l2),
    }


def candidate_id(method: str, params: dict) -> str:
    parts = [method]
    for name in sorted(params):
        value = params[name]
        if isinstance(value, float):
            value = str(value).replace(".", "p")
        parts.append(f"{name}-{value}")
    return "__".join(parts)


def evaluate_candidate(contexts: list[dict], method: str, params: dict, args) -> dict:
    records = []
    route_elapsed = 0.0
    for context in contexts:
        config = SparseHistoryConfig(
            method=method,
            history_density=args.density,
            block_size=64,
            method_params=params,
        )
        started = time.perf_counter()
        plan = route_indexed_history(
            context["summary"],
            context["indices"][int(params["remote_clusters"])],
            config,
            exact_k_tokens=0,
        )
        route_elapsed += time.perf_counter() - started
        records.append(
            {
                **teacher_metrics(
                    context["payload"], plan, sample_queries=args.sample_queries
                ),
                "history_pair_density": plan.history_pair_density,
                "history_transfer_density": plan.history_transfer_density,
                "route_plan_sha256": plan.digest(),
            }
        )
    aggregate = {}
    for field in (
        "teacher_relative_l2_mean",
        "teacher_relative_l2_p90",
        "teacher_relative_l2_max",
        "teacher_cosine_mean",
        "attention_mass_recall_mean",
        "history_pair_density",
        "history_transfer_density",
    ):
        aggregate[field] = float(torch.tensor([record[field] for record in records]).mean())
    return {
        "candidate_id": candidate_id(method, params),
        "method": method,
        "method_params": params,
        **aggregate,
        "route_elapsed_s": route_elapsed,
        "capture_records": records,
    }


def rank_key(record: dict) -> tuple:
    return (
        record["teacher_relative_l2_mean"],
        record["teacher_relative_l2_p90"],
        -record["teacher_cosine_mean"],
        record["history_transfer_density"],
        record["route_elapsed_s"],
        record["candidate_id"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--density", type=float, default=0.25)
    parser.add_argument("--sample-queries", type=int, default=18)
    parser.add_argument("--spatial-height", type=int, default=30)
    parser.add_argument("--spatial-width", type=int, default=52)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    contexts = []
    capture_provenance = []
    for value in args.capture:
        path = Path(value).resolve()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        frames = reconstruct_frames(
            payload,
            spatial_height=args.spatial_height,
            spatial_width=args.spatial_width,
        )
        query = payload["query"].detach().to("cpu")
        summary = summarize_query_for_pretransfer(query, block_size=64)
        indices = {
            clusters: build_indices(
                frames,
                clusters=clusters,
                iterations=args.iterations,
                spatial_height=args.spatial_height,
                spatial_width=args.spatial_width,
            )
            for clusters in CLUSTERS
        }
        contexts.append(
            {"path": path, "payload": payload, "summary": summary, "indices": indices}
        )
        capture_provenance.append(
            {
                "artifact_id": path.name,
                "sha256": sha256(path),
                "layer": payload.get("layer"),
                "current_start": payload.get("current_start"),
                "query_shape": list(payload["query"].shape),
                "key_shape": list(payload["key"].shape),
            }
        )

    coverage = []
    vaware = []
    for base_fraction, local_fraction in SPLITS:
        for clusters in CLUSTERS:
            common = {
                "base_fraction": base_fraction,
                "local_fraction": local_fraction,
                "remote_clusters": clusters,
                "iterations": args.iterations,
            }
            coverage.append(
                evaluate_candidate(
                    contexts, "coverage_cluster_history", common, args
                )
            )
            for v_weight in V_WEIGHTS:
                vaware.append(
                    evaluate_candidate(
                        contexts,
                        "vaware_cluster_history",
                        {**common, "v_weight": v_weight},
                        args,
                    )
                )
    coverage.sort(key=rank_key)
    vaware.sort(key=rank_key)
    best_vaware = vaware[0]["method_params"]
    transfer = [
        evaluate_candidate(
            contexts,
            "transfer_vaware_hybrid_history",
            {**best_vaware, "transfer_multiplier": multiplier},
            args,
        )
        for multiplier in TRANSFER_MULTIPLIERS
    ]
    transfer.sort(key=rank_key)
    best_error = min(item["teacher_relative_l2_mean"] for item in transfer)
    best_p90 = min(item["teacher_relative_l2_p90"] for item in transfer)
    eligible_transfer = [
        item
        for item in transfer
        if item["teacher_relative_l2_mean"] <= 1.05 * best_error
        and item["teacher_relative_l2_p90"] <= 1.10 * best_p90
    ]
    selected_transfer = min(
        eligible_transfer or transfer,
        key=lambda item: (
            item["history_transfer_density"],
            item["teacher_relative_l2_mean"],
            item["candidate_id"],
        ),
    )

    payload = {
        "artifact_id": "proposed_history_qkv_calibration_v1",
        "status": "qkv_calibrated_long_video_freeze_pending",
        "formal_prompts_used": False,
        "online_information_boundary": [
            "GPU Q block summaries",
            "CPU K prototypes",
            "CPU V prototypes",
            "frame and spatial coordinates",
        ],
        "output_residual_role": "offline_teacher_only",
        "teacher": "exact dense-history output versus sparse-history output on sampled queries",
        "captures": capture_provenance,
        "selection_rule": {
            "coverage_and_vaware": "min mean relative L2, p90 relative L2, negative cosine, transfer density, route time, candidate id",
            "transfer": "lowest transfer density within 5 percent of best mean and 10 percent of best p90 teacher error",
        },
        "qkv_selected_candidates": {
            "coverage_cluster_history": coverage[0],
            "vaware_cluster_history": vaware[0],
            "transfer_vaware_hybrid_history": selected_transfer,
        },
        "candidate_tables": {
            "coverage_cluster_history": coverage,
            "vaware_cluster_history": vaware,
            "transfer_vaware_hybrid_history": transfer,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "captures": len(contexts),
                "coverage_candidates": len(coverage),
                "vaware_candidates": len(vaware),
                "transfer_candidates": len(transfer),
                "selected": {
                    name: item["candidate_id"]
                    for name, item in payload["qkv_selected_candidates"].items()
                },
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
