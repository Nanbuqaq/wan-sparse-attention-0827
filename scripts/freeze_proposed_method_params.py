#!/usr/bin/env python3
"""Freeze proposed-route parameters after isolated QKV and long-video gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


METHODS = {
    "coverage_cluster_history",
    "vaware_cluster_history",
    "transfer_vaware_hybrid_history",
}


def artifact(path: Path) -> dict:
    return {
        "artifact_id": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-params", required=True)
    parser.add_argument("--qkv-calibration", required=True)
    parser.add_argument("--long-audit", required=True)
    parser.add_argument("--case-metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base_path = Path(args.base_params)
    qkv_path = Path(args.qkv_calibration)
    audit_path = Path(args.long_audit)
    metrics_path = Path(args.case_metrics)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    qkv = json.loads(qkv_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_path)

    if base.get("status") not in {
        "frozen_before_method_smoke",
        "frozen_before_formal_long_video",
    }:
        raise ValueError("base method parameters are not frozen")
    if qkv.get("status") != "qkv_calibrated_long_video_freeze_pending":
        raise ValueError("QKV calibration is not awaiting the long-video gate")
    if qkv.get("formal_prompts_used") is not False:
        raise ValueError("formal prompts leaked into proposed-method calibration")
    if not (
        audit.get("status") == "pass"
        and int(audit.get("expected_cases", -1)) == 8
        and int(audit.get("terminal_cases", -1)) == 8
        and not audit.get("errors")
    ):
        raise RuntimeError("isolated long-video calibration audit is incomplete")
    proposed = metrics[metrics["method"].isin(METHODS)].copy()
    counts = proposed.groupby("method")["case_id"].nunique().to_dict()
    if set(counts) != METHODS or any(int(counts[name]) != 2 for name in METHODS):
        raise RuntimeError(f"proposed calibration cases are incomplete: {counts}")
    nonpass = proposed[proposed["status"] != "pass"]
    if not nonpass.empty:
        failures = nonpass[["method", "case_id", "status"]].to_dict(orient="records")
        raise RuntimeError(
            "QKV-selected candidate failed the long-video quality gate; "
            f"run the next ranked candidate before freezing: {failures}"
        )

    selected = qkv["qkv_selected_candidates"]
    params = dict(base.get("method_params", {}))
    for method in sorted(METHODS):
        params[method] = selected[method]["method_params"]
    payload = {
        **base,
        "status": "frozen_before_formal_long_video",
        "formal_prompts_used": False,
        "method_params": params,
        "proposed_method_freeze": {
            "output_residual_role": "offline_teacher_only",
            "online_information_boundary": qkv["online_information_boundary"],
            "selected_candidates": {
                method: selected[method]["candidate_id"] for method in sorted(METHODS)
            },
            "source_artifacts": {
                "base_params": artifact(base_path),
                "qkv_calibration": artifact(qkv_path),
                "long_audit": artifact(audit_path),
                "case_metrics": artifact(metrics_path),
            },
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "methods": sorted(METHODS),
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
