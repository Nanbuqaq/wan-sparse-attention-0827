#!/usr/bin/env python3
"""Create the Dense-only human review sheet used to freeze formal prompts."""

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
    parser.add_argument("--native-states", required=True)
    parser.add_argument("--rag-states", required=True)
    parser.add_argument("--candidates", default="configs/prompts/dense_candidates.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    native = json.loads(Path(args.native_states).read_text(encoding="utf-8"))["cases"]
    rag = json.loads(Path(args.rag_states).read_text(encoding="utf-8"))["cases"]
    manifest = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    native_by = {(row["prompt_id"], int(row["seed"])): row for row in native}
    rag_by = {(row["prompt_id"], int(row["seed"])): row for row in rag}
    rows = []
    for candidate in manifest["candidates"]:
        for seed in manifest["seeds"]:
            key = (candidate["prompt_id"], int(seed))
            if key not in native_by or key not in rag_by:
                raise RuntimeError(f"missing Dense prompt-screen pair: {key}")
            native_row, rag_row = native_by[key], rag_by[key]
            for runtime, state in (
                ("native_dense", native_row),
                ("rag_dense", rag_row),
            ):
                row = {
                    "case_id": state.get("id", state.get("case_id")),
                    "commit": state.get("commit"),
                    "prompt_id": candidate["prompt_id"],
                    "category": candidate["category"],
                    "seed": seed,
                    "runtime": runtime,
                    "technical_pass": state["status"] == "pass",
                    "review_notes": "",
                    "video": state.get("video", ""),
                    "prompt": candidate["prompt"],
                }
                row.update({field: "" for field in SCORE_FIELDS})
                rows.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
