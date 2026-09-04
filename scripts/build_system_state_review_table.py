#!/usr/bin/env python3
"""Build a RAG-Dense-only review table for irreversible-state candidates."""

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
COUNT_FIELDS = ["state_reset_count", "freeze_count", "camera_cut_count"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True)
    parser.add_argument(
        "--candidates", default="configs/system/state_prompt_candidates.json"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    states = json.loads(Path(args.states).read_text(encoding="utf-8"))["cases"]
    manifest = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    by_key = {(row["prompt_id"], int(row["seed"])): row for row in states}
    rows = []
    for candidate in manifest["candidates"]:
        for seed in manifest["seeds"]:
            key = (candidate["prompt_id"], int(seed))
            if key not in by_key:
                raise RuntimeError(f"missing RAG Dense state-screen case: {key}")
            state = by_key[key]
            row = {
                "case_id": state.get("id", state.get("case_id")),
                "commit": state.get("commit"),
                "prompt_id": candidate["prompt_id"],
                "category": candidate["category"],
                "seed": int(seed),
                "runtime": "rag_dense",
                "technical_pass": state.get("status") == "pass",
                "decoded_frames": state.get("decoded_frames"),
                "video": state.get("video", ""),
                "prompt": candidate["prompt"],
                "review_notes": "",
            }
            row.update({field: "" for field in SCORE_FIELDS})
            row.update({field: "" for field in COUNT_FIELDS})
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
