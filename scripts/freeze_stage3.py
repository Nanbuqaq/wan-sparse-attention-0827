#!/usr/bin/env python3
"""Freeze Stage-3 routes after isolated 50-step metrics and visual review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from bootstrap import ROOT, configure_runtime

configure_runtime()


FAMILIES = {
    "coverage": ["coverage_b70_l15", "coverage_b80_l10"],
    "vaware": ["vaware_prototype_b80", "vaware_residual_b80"],
    "hybrid": ["hybrid_b75_r10", "hybrid_b80_r20"],
}
TARGET_METHODS = {
    "coverage": "coverage_cluster",
    "vaware": "vaware_cluster",
    "hybrid": "stage3_hybrid",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_rank(table: pd.DataFrame, column: str, maximize: bool) -> pd.Series:
    value = table[column].astype(float)
    span = value.max() - value.min()
    if span <= 1e-12:
        return pd.Series(0.0, index=table.index)
    score = (value - value.min()) / span
    return score if maximize else 1.0 - score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-suite", default="configs/stage3_calibration_50step.json")
    parser.add_argument("--metrics", default="results/metrics/stage3_calibration_50step/case_metrics.csv")
    parser.add_argument("--review", default="configs/stage3_calibration_human_review.json")
    parser.add_argument("--correctness", default="results/metrics/stage3_correctness/correctness.json")
    parser.add_argument("--kernel-benchmark", default="results/metrics/stage3_same_route_kernel.json")
    parser.add_argument("--template", default="configs/stage3_formal_50step.template.json")
    parser.add_argument("--output", default="configs/stage3_formal_50step.json")
    args = parser.parse_args()
    paths = {name: (ROOT / value) for name, value in vars(args).items() if name != "output"}
    calibration = json.loads(paths["calibration_suite"].read_text())
    review = json.loads(paths["review"].read_text())
    correctness = json.loads(paths["correctness"].read_text())
    kernel_benchmark = json.loads(paths["kernel_benchmark"].read_text())
    if review.get("status") != "COMPLETE":
        raise RuntimeError("Stage-3 human review must be COMPLETE before freezing")
    if correctness.get("status") != "pass":
        raise RuntimeError("Stage-3 route/backend correctness must pass before freezing")
    if kernel_benchmark.get("status") != "pass":
        raise RuntimeError("Stage-3 same-route kernel benchmark must pass before freezing")
    metrics = pd.read_csv(paths["metrics"])
    methods = {item["id"]: item for item in calibration["methods"]}
    selected = {}
    decision_rows = []
    for family, candidates in FAMILIES.items():
        subset = metrics[metrics["base_method_id"].isin(candidates)].copy()
        if set(subset["base_method_id"]) != set(candidates):
            raise RuntimeError(f"missing 50-step calibration metrics for {family}: {candidates}")
        if (subset["failed_calls"] != 0).any() or (subset["fallback_calls"] != 0).any():
            raise RuntimeError(f"failed/fallback calls in {family} calibration")
        if (subset["actual_density"].sub(0.25).abs() > 1e-3).any():
            raise RuntimeError(f"density mismatch in {family} calibration")
        visual = review["families"][family]
        allowed = [
            candidate
            for candidate in candidates
            if visual["candidates"][candidate]["visual_status"] in {"pass", "conditional_pass"}
            and visual["candidates"][candidate]["subject_preserved"] is True
            and visual["candidates"][candidate]["large_white_or_missing_regions"] is False
        ]
        if not allowed:
            raise RuntimeError(f"no visually usable candidate in {family}")
        subset = subset[subset["base_method_id"].isin(allowed)].set_index("base_method_id")
        score = (
            0.28 * normalized_rank(subset, "psnr_mean", True)
            + 0.14 * normalized_rank(subset, "ssim_mean", True)
            + 0.14 * normalized_rank(subset, "lpips_mean", False)
            + 0.10 * normalized_rank(subset, "flow_epe_mean", False)
            + 0.10 * normalized_rank(subset, "temporal_flicker", False)
            + 0.12 * normalized_rank(subset, "generation_elapsed_s", False)
            + 0.12 * normalized_rank(subset, "routing_p50_ms", False)
        )
        explicit = visual.get("selected_candidate")
        winner = explicit if explicit in allowed else str(score.idxmax())
        selected[family] = winner
        decision_rows.append(
            {
                "family": family,
                "selected_candidate": winner,
                "allowed_candidates": allowed,
                "composite_scores": {candidate: float(score.loc[candidate]) for candidate in score.index},
                "human_selection_used": explicit == winner,
                "selection_notes": visual.get("selection_notes", ""),
            }
        )

    template = json.loads(paths["template"].read_text())
    target_lookup = {item["id"]: item for item in template["methods"]}
    selected_backend = kernel_benchmark["selected_backend"]
    for family, candidate in selected.items():
        source = dict(methods[candidate])
        target_id = TARGET_METHODS[family]
        source["id"] = target_id
        source["result_origin"] = "stage3_new"
        source["frozen_candidate"] = candidate
        source["parameter_origin"] = f"frozen_stage3_50step_calibration:{candidate}"
        if family == "hybrid":
            source["backend"] = selected_backend["backend"]
            if selected_backend["backend"] == "varlen_triton_csr":
                source["backend_params"] = {"block_m": selected_backend["block_m"], "block_n": selected_backend["block_n"]}
            else:
                source.pop("backend_params", None)
            source.setdefault("route_params", {})["record_route_graph_hash"] = True
        target_lookup[target_id].clear()
        target_lookup[target_id].update(source)
    template["freeze_status"] = "FROZEN_STAGE3_NO_RETUNING_AFTER_FORMAL_RESULTS"
    template["frozen_on"] = "2026-08-28"
    template["frozen_candidates"] = selected
    template["calibration_evidence"] = {
        "suite": str(paths["calibration_suite"].relative_to(ROOT)),
        "suite_sha256": sha256(paths["calibration_suite"]),
        "metrics": str(paths["metrics"].relative_to(ROOT)),
        "metrics_sha256": sha256(paths["metrics"]),
        "human_review": str(paths["review"].relative_to(ROOT)),
        "human_review_sha256": sha256(paths["review"]),
        "correctness": str(paths["correctness"].relative_to(ROOT)),
        "correctness_sha256": sha256(paths["correctness"]),
        "kernel_benchmark": str(paths["kernel_benchmark"].relative_to(ROOT)),
        "kernel_benchmark_sha256": sha256(paths["kernel_benchmark"]),
        "decisions": decision_rows,
    }
    output = ROOT / args.output
    if output.exists():
        raise RuntimeError(f"refusing to overwrite frozen suite: {output}")
    output.write_text(json.dumps(template, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": template["freeze_status"], "selected": selected, "selected_backend": selected_backend, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
