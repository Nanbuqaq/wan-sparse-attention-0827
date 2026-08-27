#!/usr/bin/env python3
"""Freeze one parameter candidate per paper/self method from 50-step cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from screen_captured_qkv import candidates


QUALITY = {
    "psnr_mean": 1,
    "ssim_mean": 1,
    "lpips_mean": -1,
    "flow_epe_mean": -1,
    "temporal_flicker": -1,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-metrics",
        default="results/metrics/calibration_50step_v2/case_metrics.csv",
    )
    parser.add_argument("--human-review")
    parser.add_argument("--output", default="configs/frozen_methods_v2.json")
    args = parser.parse_args()
    table = pd.read_csv(ROOT / args.case_metrics)
    review = {}
    if args.human_review:
        review = json.loads((ROOT / args.human_review).read_text(encoding="utf-8"))
    catalog = {
        f"{item['method']}:{item['candidate']}": item for item in candidates()
    }
    groups = {}
    for method_id in sorted(table["base_method_id"].unique()):
        if "__" not in method_id:
            continue
        method, candidate = method_id.split("__", 1)
        groups.setdefault(method, []).append((method_id, candidate))
    frozen = []
    decisions = []
    case_keys = ["prompt_id", "seed", "target_density"]
    for method, entries in sorted(groups.items()):
        if len(entries) != 2:
            raise RuntimeError(f"expected two candidates for {method}, got {entries}")
        left_id, left_candidate = entries[0]
        right_id, right_candidate = entries[1]
        left = table[table["base_method_id"] == left_id]
        right = table[table["base_method_id"] == right_id]
        paired = left.merge(right, on=case_keys, suffixes=("_left", "_right"))
        left_case_wins = right_case_wins = 0
        details = []
        for row in paired.itertuples():
            left_metric_wins = right_metric_wins = 0
            for metric, direction in QUALITY.items():
                left_value = getattr(row, f"{metric}_left")
                right_value = getattr(row, f"{metric}_right")
                if direction * (left_value - right_value) > 0:
                    left_metric_wins += 1
                elif direction * (right_value - left_value) > 0:
                    right_metric_wins += 1
            if left_metric_wins >= 3:
                left_case_wins += 1
            elif right_metric_wins >= 3:
                right_case_wins += 1
            details.append(
                {
                    "prompt_id": row.prompt_id,
                    "seed": int(row.seed),
                    "density": float(row.target_density),
                    "left_metric_wins": left_metric_wins,
                    "right_metric_wins": right_metric_wins,
                }
            )
        veto = set(review.get(method, {}).get("veto_candidates", []))
        if left_candidate in veto and right_candidate in veto:
            raise RuntimeError(f"human review vetoed both candidates for {method}")
        if left_candidate in veto:
            chosen_id, chosen_candidate = right_id, right_candidate
            reason = "human_video_veto"
        elif right_candidate in veto:
            chosen_id, chosen_candidate = left_id, left_candidate
            reason = "human_video_veto"
        elif left_case_wins > right_case_wins:
            chosen_id, chosen_candidate = left_id, left_candidate
            reason = "majority_of_five_quality_metrics"
        elif right_case_wins > left_case_wins:
            chosen_id, chosen_candidate = right_id, right_candidate
            reason = "majority_of_five_quality_metrics"
        else:
            left_time = float(left["generation_elapsed_s"].mean())
            right_time = float(right["generation_elapsed_s"].mean())
            if left_time <= right_time:
                chosen_id, chosen_candidate = left_id, left_candidate
            else:
                chosen_id, chosen_candidate = right_id, right_candidate
            reason = "quality_case_tie_then_end_to_end_time"
        item = catalog[f"{method}:{chosen_candidate}"]
        frozen.append(
            {
                "id": method,
                "mode": "sparse",
                "method": method,
                "backend": "fixed64_bf16",
                "parameter_origin": "frozen_50step_calibration_v2",
                "q_clusters": item["q_clusters"],
                "k_clusters": item["k_clusters"],
                "kmeans_init_iterations": item["init_iterations"],
                "kmeans_step_iterations": item["step_iterations"],
                "route_params": item["route_params"],
                "frozen_candidate": chosen_candidate,
                "result_origin": "stage2_new",
            }
        )
        decisions.append(
            {
                "method": method,
                "left": left_candidate,
                "right": right_candidate,
                "left_case_wins": left_case_wins,
                "right_case_wins": right_case_wins,
                "chosen": chosen_candidate,
                "reason": reason,
                "case_details": details,
            }
        )
    payload = {
        "freeze_status": "FROZEN_AFTER_50STEP_CALIBRATION",
        "case_metrics": args.case_metrics,
        "methods": frozen,
        "decisions": decisions,
    }
    output = ROOT / args.output
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "methods": len(frozen)}, indent=2))


if __name__ == "__main__":
    main()

