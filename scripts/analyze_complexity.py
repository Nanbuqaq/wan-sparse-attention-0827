#!/usr/bin/env python3
"""Separate theoretical LongLive complexity from measured runtime counters."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.longlive_sparse.methods import METHOD_SPECS


def _measured_summary(path: Path) -> dict:
    stats = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path.resolve()),
        "method": stats.get("method"),
        "attention_backend": stats.get("attention_backend"),
        "calls": stats.get("calls"),
        "history_pair_density": stats.get("history_pair_density", stats.get("history_density")),
        "history_transfer_density": stats.get("history_transfer_density"),
        "global_executed_density": stats.get("global_executed_density"),
        "archive_bytes": stats.get("archive_bytes"),
        "index_bytes": stats.get("index_bytes"),
        "index_transfer_bytes": stats.get("index_transfer_bytes"),
        "query_summary_bytes": stats.get("query_summary_bytes"),
        "candidate_transfer_bytes": stats.get("candidate_transfer_bytes"),
        "transferred_bytes": stats.get("transferred_bytes"),
        "candidate_history_tokens": stats.get("candidate_history_tokens"),
        "selected_history_tokens": stats.get("selected_history_tokens"),
        "dense_qk_pairs": stats.get("dense_qk_pairs"),
        "executed_qk_pairs": stats.get("executed_qk_pairs"),
        "staging_padding_tokens": stats.get("staging_padding_tokens"),
        "timing": stats.get("timing"),
        "backend_counts": stats.get("backend_counts"),
        "routing_stage_counts": stats.get("routing_stage_counts"),
        "failed_calls": stats.get("failed_calls"),
        "dense_fallback_calls": stats.get("dense_fallback_calls"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="append", default=[])
    parser.add_argument("--cases", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--latent-frames", type=int, default=120)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layers, heads, head_dim, frame_tokens = 30, 12, 128, 1560
    query_frames, sink_frames, recent_frames, rag_candidate_frames = 3, 1, 5, 6
    dtype_bytes = 2
    query_tokens = query_frames * frame_tokens
    exact_tokens = (sink_frames + recent_frames) * frame_tokens
    history_tokens = rag_candidate_frames * frame_tokens
    dense_keys = exact_tokens + history_tokens
    kv_frame_bytes = 2 * layers * heads * frame_tokens * head_dim * dtype_bytes
    dense_history_transfer = kv_frame_bytes * rag_candidate_frames

    density_rows = []
    for density in (0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 1.00):
        history_pairs = layers * heads * query_tokens * history_tokens * density
        exact_pairs = layers * heads * query_tokens * exact_tokens
        executed_pairs = exact_pairs + history_pairs
        dense_pairs = layers * heads * query_tokens * dense_keys
        qk_pv_flops = 4 * head_dim * executed_pairs
        density_rows.append(
            {
                "history_pair_density": density,
                "global_executed_density": executed_pairs / dense_pairs,
                "executed_qk_pairs_per_forward": round(executed_pairs),
                "attention_qk_pv_flops_per_forward": round(qk_pv_flops),
                "attention_qk_pv_tflops_per_forward": qk_pv_flops / 1e12,
                "pre_transfer_selected_kv_gib_per_forward_lower_bound": dense_history_transfer * density / 2**30,
                "post_transfer_candidate_kv_gib_per_forward": dense_history_transfer / 2**30,
                "kv_hbm_read_gib_per_forward_lower_bound": 2 * layers * heads * (exact_tokens + history_tokens * density) * head_dim * dtype_bytes / 2**30,
                "fixed64_history_tokens_scheduled": math.ceil(history_tokens * density / 64) * 64,
            }
        )
    with (output_dir / "theory_density_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(density_rows[0]))
        writer.writeheader()
        writer.writerows(density_rows)

    complexity_by_method = {
        "block64_history": "O(H * Q_blocks * R * ceil(F/64) * d)",
        "kcluster32_history": "O(H * Q_blocks * R * 32 * d) after frame retrieval",
        "fixed_k128_history": "O(H * Q_groups * 128 * d)",
        "fixed_k256_history": "O(H * Q_groups * 256 * d)",
        "qlocal_kmeans8_ar": "Q-local clustering plus K-cluster scoring",
        "radius_k256_ar": "K256 scoring plus cluster-radius bound",
        "qmetric_k256_r32_ar": "rank-32 query covariance plus K256 scoring",
        "temporal_k256_t16_ar": "16 temporal bins plus per-bin K clustering",
        "sizesplit_k128_c2_ar": "K128 plus capacity-triggered recursive splits",
        "svg2_ar": "Q*K centroid scoring after Q/K clustering",
        "adacluster_ar": "threshold-controlled adaptive Q/K clustering",
        "svoo_ar": "iterative Q/K co-clustering and centroid scoring",
        "scope_ar": "three key-subspace cluster tables plus Q-cluster lookup",
        "coverage_cluster_history": "GPU Q-block summaries plus CPU K-cluster/block prototype coverage routing",
        "vaware_cluster_history": "coverage routing plus probability-proxy weighted CPU V prototypes",
        "transfer_vaware_hybrid_history": "V-aware coverage routing plus a bounded shared history union",
    }
    method_rows = []
    for name, spec in METHOD_SPECS.items():
        if spec.routing_stage == "N/A":
            transfer = "N/A"
        elif spec.routing_stage == "pre-transfer":
            transfer = "selected unique KV plus staging padding"
        else:
            transfer = "complete RAG frame-candidate KV at route refresh"
        method_rows.append(
            {
                "method": name,
                "category": spec.category,
                "routing_stage": spec.routing_stage,
                "routing_complexity": complexity_by_method.get(name, "baseline/no fine clustering"),
                "cpu_to_gpu_transfer": transfer,
                "parameter_origin": spec.parameter_origin,
            }
        )
    with (output_dir / "theory_method_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(method_rows[0]))
        writer.writeheader()
        writer.writerows(method_rows)

    archive_frames = max(0, args.latent_frames - 12)
    measured_cases = []
    for value in args.cases:
        payload = json.loads(Path(value).read_text(encoding="utf-8"))
        for case in payload["cases"]:
            measured_cases.append(
                {
                    key: case.get(key)
                    for key in (
                        "id",
                        "commit",
                        "method",
                        "prompt_id",
                        "seed",
                        "latent_frames",
                        "status",
                        "end_to_end_s",
                        "model_load_s_total",
                        "model_load_s_amortized",
                        "end_to_end_with_amortized_load_s",
                        "peak_allocated_gb",
                        "routing_s",
                        "q_summary_s",
                        "d2h_s",
                        "cpu_gather_s",
                        "h2d_s",
                        "rope_s",
                        "attention_s",
                        "archive_bytes",
                        "index_bytes",
                        "index_transfer_bytes",
                        "query_summary_bytes",
                        "candidate_transfer_bytes",
                        "transferred_bytes",
                        "staging_padding_tokens",
                    )
                }
            )
    summary = {
        "theory": {
            "shape": {
                "layers": layers,
                "heads": heads,
                "head_dim": head_dim,
                "frame_tokens": frame_tokens,
                "query_tokens": query_tokens,
                "exact_tokens": exact_tokens,
                "rag_candidate_history_tokens": history_tokens,
            },
            "kv_bytes_per_latent_frame_all_layers": kv_frame_bytes,
            "kv_mib_per_latent_frame_all_layers": kv_frame_bytes / 2**20,
            "gpu_12_frame_cache_gib": kv_frame_bytes * 12 / 2**30,
            "dense_6_frame_transfer_gib_per_forward": dense_history_transfer / 2**30,
            "archive_frames_at_video_end": archive_frames,
            "cpu_archive_gib_at_video_end": kv_frame_bytes * archive_frames / 2**30,
            "frame_retrieval_growth": "O(T_history * descriptor_dim) per refresh for exact scan",
            "archive_capacity_growth": "O(T_history * layers * heads * frame_tokens * head_dim)",
        },
        "measured": [_measured_summary(Path(value)) for value in args.stats],
        "measured_cases": measured_cases,
        "separation_rule": "theory and measured counters are separate fields; theoretical values are never labelled measured",
    }
    (output_dir / "complexity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
