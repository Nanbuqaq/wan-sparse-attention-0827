#!/usr/bin/env python3
"""Cross-check the final frozen Pareto result bundle and its audit chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminal", required=True)
    parser.add_argument("--recovery", required=True)
    parser.add_argument("--quality", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--long-diagnostics", required=True)
    parser.add_argument("--training-gate", required=True)
    parser.add_argument("--video-index", required=True)
    parser.add_argument("--figures-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = {
        name: Path(getattr(args, name.replace("-", "_"))).resolve()
        for name in (
            "terminal",
            "recovery",
            "quality",
            "summary",
            "long_diagnostics",
            "training_gate",
            "video_index",
        )
    }
    errors = []
    terminal = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    if (
        terminal.get("status") != "pass"
        or terminal.get("expected_cases") != 102
        or terminal.get("terminal_cases") != 102
        or terminal.get("pass_cases") != 100
        or terminal.get("fail_cases") != 2
        or terminal.get("negative_cases") != 0
        or terminal.get("errors")
    ):
        errors.append("terminal audit does not match 100 pass + 2 preserved fail")

    recovery = json.loads(paths["recovery"].read_text(encoding="utf-8"))
    if (
        recovery.get("status") != "pass"
        or recovery.get("cases") != 102
        or recovery.get("successful_cases") != 100
        or recovery.get("artifact_records") != 400
        or recovery.get("local_audit_status") != "pass"
    ):
        errors.append("local artifact recovery audit is incomplete")

    quality = json.loads(paths["quality"].read_text(encoding="utf-8"))
    if quality.get("status") != "pass" or len(quality.get("groups", [])) != 12:
        errors.append("quality manifest is incomplete")
    quality_candidates = 0
    lpips_statuses = set()
    for group in quality.get("groups", []):
        summary_path = Path(group["summary"])
        group_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        quality_candidates += len(group_summary.get("candidates", {}))
        lpips_statuses.update(
            item.get("lpips_status")
            for item in group_summary.get("candidates", {}).values()
        )
    if quality_candidates != 100 or lpips_statuses != {"available"}:
        errors.append("quality metrics are missing candidates or audited LPIPS")

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    if summary.get("status") != "pass" or summary.get("cases") != 102:
        errors.append("Pareto summary is incomplete")
    expected_pareto = {
        "fixed_k256_history",
        "transfer_vaware_hybrid_history",
    }
    if set(summary.get("system_pareto_methods", [])) != expected_pareto:
        errors.append("system Pareto method set changed")

    long_diagnostics = json.loads(
        paths["long_diagnostics"].read_text(encoding="utf-8")
    )
    long_cases = long_diagnostics.get("cases", [])
    if len(long_cases) != 16:
        errors.append("long-video diagnostic set must contain 16 videos")
    if any(
        case.get("freeze_runs")
        or case.get("cut_indices")
        or case.get("flicker_indices")
        for case in long_cases
    ):
        errors.append("long-video automatic diagnostics contain anomalies")

    training = json.loads(paths["training_gate"].read_text(encoding="utf-8"))
    if (
        training.get("status") != "pass"
        or training.get("decision") != "do_not_train"
        or training.get("training_triggered") is not False
    ):
        errors.append("training gate did not freeze do_not_train")

    video_index = pd.read_csv(paths["video_index"])
    if len(video_index) != 102 or int(video_index["video"].notna().sum()) != 100:
        errors.append("video index does not contain 102 cases / 100 videos")

    figures_dir = Path(args.figures_dir).resolve()
    figures = sorted(figures_dir.glob("*.png"))
    if len(figures) != 3 or any(path.stat().st_size == 0 for path in figures):
        errors.append("expected three non-empty Pareto figures")

    payload = {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "terminal_cases": terminal.get("terminal_cases"),
        "pass_cases": terminal.get("pass_cases"),
        "fail_cases": terminal.get("fail_cases"),
        "recovered_artifacts": recovery.get("artifact_records"),
        "recovered_bytes": recovery.get("artifact_bytes"),
        "quality_groups": len(quality.get("groups", [])),
        "quality_candidates": quality_candidates,
        "lpips_statuses": sorted(lpips_statuses),
        "long_videos": len(long_cases),
        "system_pareto_methods": sorted(expected_pareto),
        "training_decision": training.get("decision"),
        "figures": [str(path) for path in figures],
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
