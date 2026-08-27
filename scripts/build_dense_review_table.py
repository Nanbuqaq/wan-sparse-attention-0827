#!/usr/bin/env python3
"""Create the Dense-only human review sheet used to freeze formal prompts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


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
            rows.append(
                {
                    "prompt_id": candidate["prompt_id"],
                    "category": candidate["category"],
                    "seed": seed,
                    "native_dense_pass": native_row["status"] == "pass",
                    "rag_dense_pass": rag_row["status"] == "pass",
                    "dense_quality_score": "",
                    "identity_consistency": "",
                    "background_consistency": "",
                    "state_progression": "",
                    "action_continuity": "",
                    "freeze_or_camera_cut": "",
                    "review_notes": "",
                    "native_video": native_row.get("video", ""),
                    "rag_video": rag_row.get("video", ""),
                    "prompt": candidate["prompt"],
                }
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
