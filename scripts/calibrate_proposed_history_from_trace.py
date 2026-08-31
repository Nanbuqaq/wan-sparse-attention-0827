#!/usr/bin/env python3
"""Calibrate proposed pre-transfer routes with exact offline output teachers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
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
PROTOTYPE_BLOCK_SIZE = 64
QUERY_BLOCK_SIZES = (64, 128, 256)
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
    spatial_height: int,
    spatial_width: int,
) -> list:
    config = SparseHistoryConfig(
        method="coverage_cluster_history",
        history_density=0.25,
        block_size=PROTOTYPE_BLOCK_SIZE,
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


def candidate_coordinate_index(payload: dict) -> dict:
    """Build a batched exact coordinate-to-candidate-index lookup."""

    frame_ids = payload["frame_ids"].detach().to("cpu").long()
    token_ids = payload["token_ids"].detach().to("cpu").long()
    if frame_ids.shape != token_ids.shape or frame_ids.ndim != 3:
        raise ValueError("capture frame/token ids must share [B,H,K]")
    coordinate_base = int(token_ids.max()) + 1
    codes = frame_ids * coordinate_base + token_ids
    sorted_codes, sorted_to_dense = torch.sort(codes, dim=-1)
    if sorted_codes.shape[-1] > 1 and bool(
        (sorted_codes[..., 1:] == sorted_codes[..., :-1]).any()
    ):
        raise ValueError("capture candidate coordinates are not unique")
    return {
        "coordinate_base": coordinate_base,
        "sorted_codes": sorted_codes,
        "sorted_to_dense": sorted_to_dense,
    }


def prepare_teacher(payload: dict, *, sample_queries: int) -> dict:
    """Precompute exact Dense teacher tensors once for one QKV capture."""

    query = payload["query"].detach().to("cpu").float()
    key = payload["key"].detach().to("cpu").float()
    value = payload["value"].detach().to("cpu").float()
    coordinate_index = candidate_coordinate_index(payload)
    batch, query_tokens, heads, dim = query.shape
    sample_ids = torch.linspace(0, query_tokens - 1, sample_queries).round().long()
    sampled_query = query.index_select(1, sample_ids).permute(0, 2, 1, 3)
    dense_key = key.permute(0, 2, 1, 3)
    dense_value = value.permute(0, 2, 1, 3)
    dense_logits = torch.einsum(
        "bhsd,bhkd->bhsk", sampled_query, dense_key
    ) / math.sqrt(dim)
    dense_probability = torch.softmax(dense_logits, dim=-1)
    dense_output = torch.einsum(
        "bhsk,bhkd->bhsd", dense_probability, dense_value
    )
    return {
        **coordinate_index,
        "sample_ids": sample_ids,
        "dense_value": dense_value,
        "dense_logits": dense_logits,
        "dense_probability": dense_probability,
        "dense_output": dense_output,
    }


def selected_candidate_indices(teacher: dict, plan) -> torch.Tensor:
    """Map a route plan to dense candidate indices without Python token loops."""

    union_frames = plan.union_frame_ids.long()
    union_tokens = plan.union_token_ids.long()
    valid_union = union_frames >= 0
    union_codes = (
        union_frames * int(teacher["coordinate_base"]) + union_tokens.clamp_min(0)
    )
    sorted_codes = teacher["sorted_codes"]
    sorted_to_dense = teacher["sorted_to_dense"]
    positions = torch.searchsorted(
        sorted_codes.contiguous(), union_codes.contiguous()
    ).clamp_max(sorted_codes.shape[-1] - 1)
    union_to_dense = sorted_to_dense.gather(-1, positions)
    matched = sorted_codes.gather(-1, positions) == union_codes
    if not bool((matched | ~valid_union).all()):
        raise KeyError("route plan contains coordinates outside the captured candidate KV")

    sample_ids = teacher["sample_ids"]
    sample_groups = plan.query_labels.index_select(2, sample_ids)
    sample_counts = plan.group_history_counts.gather(2, sample_groups)
    selected_width = int(sample_counts.max())
    if not bool((sample_counts == selected_width).all()):
        raise ValueError("offline teacher requires one exact budget per sampled group")
    group_index = sample_groups.unsqueeze(-1).expand(
        -1, -1, -1, plan.group_union_indices.shape[-1]
    )
    sample_union = plan.group_union_indices.gather(2, group_index)[
        ..., :selected_width
    ]
    if not bool((sample_union >= 0).all()):
        raise ValueError("sampled route group contains padded union indices")
    union_lookup = union_to_dense.unsqueeze(2).expand(
        -1, -1, sample_union.shape[2], -1
    )
    return union_lookup.gather(3, sample_union)


def teacher_metrics(teacher: dict, plan) -> dict:
    selected = selected_candidate_indices(teacher, plan)
    dense_value = teacher["dense_value"]
    dense_probability = teacher["dense_probability"]
    dense_output = teacher["dense_output"]
    sparse_logits = teacher["dense_logits"].gather(-1, selected)
    sparse_probability = torch.softmax(sparse_logits, dim=-1)
    value_index = selected.unsqueeze(-1).expand(
        -1, -1, -1, -1, dense_value.shape[-1]
    )
    selected_value = dense_value.unsqueeze(2).expand(
        -1, -1, selected.shape[2], -1, -1
    ).gather(3, value_index)
    sparse_output = torch.einsum(
        "bhsk,bhskd->bhsd", sparse_probability, selected_value
    )
    relative_l2 = (
        torch.linalg.vector_norm(sparse_output - dense_output, dim=-1)
        / torch.linalg.vector_norm(dense_output, dim=-1).clamp_min(1e-8)
    )
    cosines = F.cosine_similarity(sparse_output, dense_output, dim=-1)
    mass_recalls = dense_probability.gather(-1, selected).sum(dim=-1)
    values = relative_l2.reshape(-1)
    return {
        "teacher_relative_l2_mean": float(values.mean()),
        "teacher_relative_l2_p90": float(torch.quantile(values, 0.90)),
        "teacher_relative_l2_max": float(values.max()),
        "teacher_cosine_mean": float(cosines.mean()),
        "attention_mass_recall_mean": float(mass_recalls.mean()),
        "teacher_queries": values.numel(),
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
            context["summaries"][int(params["query_block_size"])],
            context["indices"],
            config,
            exact_k_tokens=0,
        )
        route_elapsed += time.perf_counter() - started
        records.append(
            {
                **teacher_metrics(
                    context["teacher"], plan
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
        summaries = {
            block_size: summarize_query_for_pretransfer(
                query, block_size=block_size
            )
            for block_size in QUERY_BLOCK_SIZES
        }
        indices = build_indices(
            frames,
            spatial_height=args.spatial_height,
            spatial_width=args.spatial_width,
        )
        contexts.append(
            {
                "path": path,
                "teacher": prepare_teacher(
                    payload, sample_queries=args.sample_queries
                ),
                "summaries": summaries,
                "indices": indices,
            }
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
        for query_block_size in QUERY_BLOCK_SIZES:
            common = {
                "base_fraction": base_fraction,
                "local_fraction": local_fraction,
                "query_block_size": query_block_size,
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
        "artifact_id": "proposed_history_qkv_calibration_v2",
        "status": "qkv_calibrated_long_video_freeze_pending",
        "analysis_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "analysis_worktree_clean": not bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        ),
        "formal_prompts_used": False,
        "online_information_boundary": [
            "GPU Q block summaries",
            "CPU Block64 K mean prototypes",
            "CPU Block64 V mean prototypes",
            "frame and spatial coordinates",
        ],
        "output_residual_role": "offline_teacher_only",
        "remote_prototype_policy": {
            "name": "block64_kv_mean",
            "block_size": PROTOTYPE_BLOCK_SIZE,
            "prototypes_per_1560_token_frame": math.ceil(
                1560 / PROTOTYPE_BLOCK_SIZE
            ),
            "token_kmeans": False,
        },
        "query_summary_block_size_candidates": list(QUERY_BLOCK_SIZES),
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
