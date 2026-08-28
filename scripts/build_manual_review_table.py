#!/usr/bin/env python3
"""Build a complete-video manual consistency/reset review sheet."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "case_id",
    "case_key_sha256",
    "commit",
    "method",
    "routing_stage",
    "prompt_id",
    "seed",
    "video",
    "status",
    "subject_identity_1to5",
    "background_consistency_1to5",
    "irreversible_state_reset_count",
    "action_loop_count",
    "action_discontinuity_count",
    "freeze_count",
    "flicker_count",
    "camera_cut_count",
    "late_quarter_quality_1to5",
    "late_quarter_degradation_0to2",
    "reviewer",
    "review_notes",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for state_path in args.states:
        payload = json.loads(Path(state_path).read_text(encoding="utf-8"))
        for case in payload["cases"]:
            if case.get("status") not in {"pass", "negative"}:
                continue
            rows.append(
                {
                    "case_id": case.get("id", case.get("case_id")),
                    "case_key_sha256": case.get("case_key_sha256"),
                    "commit": case.get("commit"),
                    "method": case.get("method", case.get("runtime")),
                    "routing_stage": case.get("routing_stage"),
                    "prompt_id": case.get("prompt_id"),
                    "seed": case.get("seed"),
                    "video": case.get("video"),
                    "status": case.get("status"),
                    "subject_identity_1to5": "",
                    "background_consistency_1to5": "",
                    "irreversible_state_reset_count": "",
                    "action_loop_count": "",
                    "action_discontinuity_count": "",
                    "freeze_count": "",
                    "flicker_count": "",
                    "camera_cut_count": "",
                    "late_quarter_quality_1to5": "",
                    "late_quarter_degradation_0to2": "",
                    "reviewer": "",
                    "review_notes": "",
                }
            )
    rows.sort(key=lambda row: (str(row["method"]), str(row["prompt_id"]), int(row["seed"])))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
