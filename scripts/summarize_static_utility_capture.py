#!/usr/bin/env python3
"""Rank static online utility candidates from isolated capture evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(paths: list[Path]) -> dict:
    if not paths:
        raise ValueError("static utility summary requires input artifacts")
    rows = []
    legacy_rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "pass" or payload.get("mode") != "utility":
            raise ValueError(f"invalid utility artifact: {path}")
        if payload.get("marginal_candidates_evaluated"):
            raise ValueError("static-only summary cannot mix marginal candidates")
        for candidate, record in payload["records"].items():
            target = legacy_rows if candidate == "legacy_final_reference" else rows
            if not candidate.endswith("__static_block"):
                if candidate != "legacy_final_reference":
                    raise ValueError(f"unexpected non-static candidate: {candidate}")
            target.append(
                {
                    "artifact": str(path),
                    "layer": int(payload["capture_metadata"]["layer"]),
                    "current_start": int(payload["capture_metadata"]["current_start"]),
                    "candidate": candidate,
                    "relative_l2": float(
                        record["history_only_output_error"]["relative_l2"]
                    ),
                    "one_minus_cosine": float(
                        record["history_only_output_error"]["one_minus_cosine"]
                    ),
                    "history_pair_density": float(
                        record["route"]["history_pair_density"]
                    ),
                    "history_transfer_density": float(
                        record["route"]["history_transfer_density"]
                    ),
                    "online_route_s": float(record["online_route_s"]),
                    "copied_bytes": int(
                        record["transfer_execution"]["copied_bytes"]
                    ),
                }
            )
    candidates = sorted(set(row["candidate"] for row in rows))
    if len(legacy_rows) != len(paths):
        raise ValueError("each utility artifact must include legacy Final reference")
    legacy_by_artifact = {row["artifact"]: row for row in legacy_rows}
    aggregate = {}
    for candidate in candidates:
        selected = [row for row in rows if row["candidate"] == candidate]
        aggregate[candidate] = {
            "cases": len(selected),
            "worst_relative_l2": max(row["relative_l2"] for row in selected),
            "worst_one_minus_cosine": max(
                row["one_minus_cosine"] for row in selected
            ),
            "worst_online_route_s": max(row["online_route_s"] for row in selected),
            "mean_history_pair_density": sum(
                row["history_pair_density"] for row in selected
            )
            / len(selected),
            "mean_history_transfer_density": sum(
                row["history_transfer_density"] for row in selected
            )
            / len(selected),
            "worst_relative_l2_delta_vs_legacy": max(
                row["relative_l2"]
                - legacy_by_artifact[row["artifact"]]["relative_l2"]
                for row in selected
            ),
            "not_worse_than_legacy_all_captures": all(
                row["relative_l2"]
                <= legacy_by_artifact[row["artifact"]]["relative_l2"]
                for row in selected
            ),
        }
    order = sorted(
        candidates,
        key=lambda candidate: (
            aggregate[candidate]["worst_relative_l2_delta_vs_legacy"],
            aggregate[candidate]["worst_relative_l2"],
            aggregate[candidate]["worst_one_minus_cosine"],
            aggregate[candidate]["worst_online_route_s"],
        ),
    )
    return {
        "status": "pass",
        "artifacts": len(paths),
        "aggregate": aggregate,
        "legacy_final_reference": {
            "cases": len(legacy_rows),
            "worst_relative_l2": max(row["relative_l2"] for row in legacy_rows),
            "worst_one_minus_cosine": max(
                row["one_minus_cosine"] for row in legacy_rows
            ),
        },
        "quality_first_order": order,
        "retained_for_motion_state_calibration": order[:2],
        "final_utility_frozen": False,
        "marginal_cost_candidates_status": "stopped_after_heldout_cost_mape_gate",
        "evidence_limits": [
            "teacher covers history K/V only and omits exact/current K/V",
            "captures do not replace isolated motion/state video calibration",
        ],
        "rows": rows,
        "legacy_rows": legacy_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = summarize([Path(value) for value in args.input])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "retained": payload["retained_for_motion_state_calibration"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
