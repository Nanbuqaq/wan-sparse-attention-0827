#!/usr/bin/env python3
"""Aggregate query-policy capture evidence without over-promoting it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(paths: list[Path]) -> dict:
    if not paths:
        raise ValueError("query-policy summary requires input artifacts")
    rows = []
    transfer_invariance = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "pass" or payload.get("mode") != "query_policy":
            raise ValueError(f"invalid query-policy artifact: {path}")
        records = payload["records"]
        copied_bytes = {
            name: int(record["transfer_execution"]["copied_bytes"])
            for name, record in records.items()
        }
        transfer_invariance.append(len(set(copied_bytes.values())) == 1)
        for policy, record in records.items():
            rows.append(
                {
                    "artifact": str(path),
                    "layer": int(payload["capture_metadata"]["layer"]),
                    "current_start": int(payload["capture_metadata"]["current_start"]),
                    "policy": policy,
                    "history_pair_density": float(
                        record["route"]["history_pair_density"]
                    ),
                    "history_transfer_density": float(
                        record["route"]["history_transfer_density"]
                    ),
                    "relative_l2": float(
                        record["history_only_output_error"]["relative_l2"]
                    ),
                    "one_minus_cosine": float(
                        record["history_only_output_error"]["one_minus_cosine"]
                    ),
                    "copied_bytes": int(
                        record["transfer_execution"]["copied_bytes"]
                    ),
                }
            )
    policies = sorted(set(row["policy"] for row in rows))
    aggregate = {}
    for policy in policies:
        selected = [row for row in rows if row["policy"] == policy]
        aggregate[policy] = {
            "cases": len(selected),
            "worst_relative_l2": max(row["relative_l2"] for row in selected),
            "worst_one_minus_cosine": max(
                row["one_minus_cosine"] for row in selected
            ),
            "mean_history_pair_density": sum(
                row["history_pair_density"] for row in selected
            )
            / len(selected),
            "mean_history_transfer_density": sum(
                row["history_transfer_density"] for row in selected
            )
            / len(selected),
        }
    quality_order = sorted(
        policies,
        key=lambda policy: (
            aggregate[policy]["worst_relative_l2"],
            aggregate[policy]["worst_one_minus_cosine"],
            aggregate[policy]["mean_history_pair_density"],
        ),
    )
    return {
        "status": "pass",
        "artifacts": len(paths),
        "rows": rows,
        "aggregate": aggregate,
        "physical_transfer_invariant_all_cases": all(transfer_invariance),
        "quality_first_order": quality_order,
        "preliminary_capture_winner": quality_order[0],
        "final_policy_frozen": False,
        "retained_for_motion_state_calibration": [
            "legacy_exact_union",
            "top_p_095",
        ],
        "evidence_limits": [
            "capture teacher covers history K/V only and omits exact/current K/V",
            "the four captures are not the isolated motion/state calibration videos",
            "capture ranking cannot replace long-video subject/background/state review",
        ],
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
                "winner": payload["preliminary_capture_winner"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
