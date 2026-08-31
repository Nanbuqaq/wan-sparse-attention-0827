#!/usr/bin/env python3
"""Evaluate proposed calibration videos against matched RAG Dense references."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "coverage_cluster_history",
    "vaware_cluster_history",
    "transfer_vaware_hybrid_history",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    states = json.loads(Path(args.states).read_text(encoding="utf-8"))["cases"]
    output = Path(args.output_dir).resolve()
    summaries = []
    for prompt_id in sorted({case["prompt_id"] for case in states}):
        rows = [case for case in states if case["prompt_id"] == prompt_id]
        dense = [case for case in rows if case["method"] == "rag_dense"]
        sparse = [case for case in rows if case["method"] in METHODS]
        if len(dense) != 1 or len(sparse) != 3:
            raise RuntimeError(
                f"{prompt_id}: expected one RAG Dense and three proposed cases"
            )
        if any(case["status"] != "pass" for case in rows):
            raise RuntimeError(f"{prompt_id}: calibration contains a non-pass case")
        prompt_output = output / prompt_id
        command = [
            sys.executable,
            str(ROOT / "scripts/evaluate_videos.py"),
            "--reference",
            dense[0]["video"],
            "--output-dir",
            str(prompt_output),
            "--no-lpips",
        ]
        for case in sorted(sparse, key=lambda item: item["method"]):
            command.extend(["--candidate", f"{case['id']}={case['video']}"])
        subprocess.run(command, cwd=ROOT, check=True)
        summaries.append(str(prompt_output / "paired_video_summary.json"))
    manifest = {"status": "pass", "quality_summaries": summaries}
    output.mkdir(parents=True, exist_ok=True)
    (output / "quality_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
