#!/usr/bin/env python3
"""Apply an audited Dense human-review score manifest to the review table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SCORE_FIELDS = [
    "category_completion_0to2",
    "subject_consistency_0to2",
    "background_consistency_0to2",
    "continuous_motion_0to2",
    "freeze_flicker_cut_0to2",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.review).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    payload = json.loads(Path(args.scores).read_text(encoding="utf-8"))
    scores = payload.get("cases")
    decisions = payload.get("decisions")
    if (scores is None) == (decisions is None):
        raise ValueError("score manifest requires exactly one of cases or decisions")
    review_ids = {row["case_id"] for row in rows}
    if scores is not None and review_ids != set(scores):
        missing = sorted(review_ids - set(scores))
        extra = sorted(set(scores) - review_ids)
        raise RuntimeError(f"score manifest mismatch: missing={missing}, extra={extra}")

    for row in rows:
        if scores is not None:
            decision = scores[row["case_id"]]
        else:
            matched = [
                item
                for item in decisions
                if item.get("prompt_id", row["prompt_id"]) == row["prompt_id"]
                and item.get("runtime", row["runtime"]) == row["runtime"]
                and int(item.get("seed", row["seed"])) == int(row["seed"])
            ]
            if not matched:
                raise RuntimeError(f"no score decision matches {row['case_id']}")
            decision = {}
            notes = []
            for item in matched:
                decision.update({field: item[field] for field in SCORE_FIELDS if field in item})
                if item.get("review_notes"):
                    notes.append(str(item["review_notes"]))
            decision["review_notes"] = " ".join(notes)
            missing_fields = set(SCORE_FIELDS) - set(decision)
            if missing_fields:
                raise RuntimeError(
                    f"incomplete score decision for {row['case_id']}: {sorted(missing_fields)}"
                )
        for field in SCORE_FIELDS:
            value = int(decision[field])
            if value not in {0, 1, 2}:
                raise ValueError(f"{row['case_id']}: invalid {field}={value}")
            row[field] = value
        row["review_notes"] = str(decision["review_notes"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
