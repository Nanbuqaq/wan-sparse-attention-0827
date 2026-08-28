#!/usr/bin/env python3
"""Final independent audit for Stage-3 suites, metrics, videos, and evidence."""

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

from run_matrix import expand_tasks, resolve_common


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="configs/stage3_formal_50step.json")
    args = parser.parse_args()
    suite_path = ROOT / args.suite
    suite = load(suite_path)
    suite["common"] = resolve_common(suite["common"])
    tasks = expand_tasks(suite)
    task_rows = []
    missing = []
    for task in tasks:
        video = ROOT / task["output"]
        stats_path = video.with_suffix(".stats.json")
        if not video.is_file() or not stats_path.is_file():
            missing.append({"task": task["id"], "video": str(video), "stats": str(stats_path)})
            continue
        stats = load(stats_path)
        sparse = stats.get("sparse") or {}
        task_rows.append(
            {
                "matrix_id": task["matrix_id"],
                "prompt_id": task["prompt_id"],
                "seed": task["seed"],
                "method_id": task["base_method_id"],
                "mode": task["mode"],
                "status": stats.get("status"),
                "result_origin": stats.get("result_origin", task.get("result_origin")),
                "failed_calls": sparse.get("failed_calls", 0),
                "fallback_calls": sparse.get("dense_fallback_calls", 0),
                "actual_density": sparse.get("logical_pair_density"),
                "video_sha256": stats.get("output_sha256"),
            }
        )
    correctness_path = ROOT / "results/metrics/stage3_correctness/correctness.json"
    qkv_path = ROOT / "results/metrics/stage3_qkv_diagnostics/summary.json"
    cpu_qkv_path = ROOT / "results/metrics/stage3_qkv_objectives_cpu/summary.json"
    calibration_audit_path = ROOT / "results/metrics/stage3_calibration_50step/evaluation_audit.json"
    formal_audit_path = ROOT / "results/metrics/stage3_formal_50step/evaluation_audit.json"
    backend_audit_path = ROOT / "results/metrics/stage3_backend_100_50step/evaluation_audit.json"
    comparison_path = ROOT / "results/manifests/stage3/comparison_videos.json"
    review_path = ROOT / "configs/stage3_formal_human_review.json"
    case_path = ROOT / "results/metrics/stage3_formal_50step/case_metrics.csv"
    method_path = ROOT / "results/metrics/stage3_formal_50step/stage3_method_table.csv"
    case_table = pd.read_csv(case_path) if case_path.is_file() else pd.DataFrame()
    sparse_rows = [row for row in task_rows if row["mode"] == "sparse"]
    checks = {
        "formal_suite_frozen": suite.get("freeze_status") == "FROZEN_STAGE3_NO_RETUNING_AFTER_FORMAL_RESULTS",
        "expected_tasks_complete": len(task_rows) == len(tasks) and not missing and all(row["status"] == "completed" for row in task_rows),
        "failed_calls_zero": all(row["failed_calls"] == 0 for row in sparse_rows),
        "fallback_calls_zero": all(row["fallback_calls"] == 0 for row in sparse_rows),
        "density_exact": all(row["actual_density"] is not None and abs(float(row["actual_density"]) - 0.25) <= 1e-3 for row in sparse_rows),
        "correctness_pass": correctness_path.is_file() and load(correctness_path).get("status") == "pass",
        "captured_route_diagnostics_pass": qkv_path.is_file() and load(qkv_path).get("status") == "pass",
        "captured_v_objectives_pass": cpu_qkv_path.is_file() and load(cpu_qkv_path).get("status") == "pass",
        "calibration_evaluation_pass": calibration_audit_path.is_file() and load(calibration_audit_path).get("status") == "pass",
        "formal_evaluation_pass": formal_audit_path.is_file() and load(formal_audit_path).get("status") == "pass",
        "backend_100_evaluation_pass": backend_audit_path.is_file() and load(backend_audit_path).get("status") == "pass",
        "comparison_videos_pass": comparison_path.is_file() and load(comparison_path).get("status") == "pass",
        "formal_human_review_complete": review_path.is_file() and load(review_path).get("status") == "COMPLETE",
        "method_table_present": method_path.is_file(),
        "case_statistics_present": (ROOT / "results/metrics/stage3_formal_50step/stage3_case_statistics_vs_block.csv").is_file(),
        "quality_speed_figure_present": (ROOT / "results/figures/stage3_formal_50step/quality_speed_pareto.png").is_file(),
        "final_report_present": (ROOT / "docs/FINAL_REPORT_STAGE3.md").is_file(),
        "stage3_manifest_present": (ROOT / "results/manifests/stage3_manifest.json").is_file() and (ROOT / "results/manifests/STAGE3_SHA256SUMS.txt").is_file(),
        "interrupted_failure_preserved": any((ROOT / "results/videos/stage3_formal_50step").rglob("*.attempt_*.error.json")),
        "four_prompt_main_panel": not case_table.empty and set(case_table[case_table["matrix_id"] == "stage3_main_d250"]["prompt_id"]) == {"gymnast_ribbon", "skateboard_alley", "koi_reflections", "orchestra_conductor"},
        "second_seed_present": not case_table.empty and not case_table[case_table["matrix_id"] == "stage3_second_seed_d250"].empty,
        "negative_cases_present": not case_table.empty and set(case_table[case_table["matrix_id"] == "stage3_negative_d250"]["prompt_id"]) == {"fox_snow", "glassblower"},
    }
    payload = {
        "schema_version": 3,
        "status": "pass" if all(checks.values()) else "incomplete",
        "checks": checks,
        "counts": {"expected_tasks": len(tasks), "completed_task_records": len(task_rows), "sparse_tasks": len(sparse_rows), "case_rows": len(case_table)},
        "missing": missing,
        "task_rows": task_rows,
    }
    output = ROOT / "results/manifests/final_audit_stage3.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    public = ROOT / "results/manifests/public/final_audit_stage3.json"
    public.write_text(json.dumps({"schema_version": 3, "status": payload["status"], "checks": checks, "counts": payload["counts"]}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "checks_passed": sum(checks.values()), "checks_total": len(checks), "output": str(output)}, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
