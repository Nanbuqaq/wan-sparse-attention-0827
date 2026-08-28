#!/usr/bin/env python3
"""Build Stage-3 50-step contact sheets and a human-review template."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import av
import cv2
import numpy as np

from run_matrix import expand_tasks, resolve_common


FAMILIES = {
    "coverage": ["coverage_b70_l15", "coverage_b80_l10"],
    "vaware": ["vaware_prototype_b80", "vaware_residual_b80"],
    "hybrid": ["hybrid_b75_r10", "hybrid_b80_r20"],
}


def decode(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        rows = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    if len(rows) != 81:
        raise RuntimeError(f"expected 81 frames, got {len(rows)}: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="configs/stage3_calibration_50step.json")
    args = parser.parse_args()
    suite_path = Path(args.suite).resolve()
    suite = json.loads(suite_path.read_text())
    suite["common"] = resolve_common(suite["common"])
    completed = {}
    for task in expand_tasks(suite):
        video = ROOT / task["output"]
        stats = video.with_suffix(".stats.json")
        if video.is_file() and stats.is_file() and json.loads(stats.read_text()).get("status") == "completed":
            completed[task["base_method_id"]] = video
    if "dense" not in completed:
        raise RuntimeError("missing Stage-3 calibration Dense video")
    output_dir = ROOT / "results/figures/stage3_calibration_review"
    output_dir.mkdir(parents=True, exist_ok=True)
    dense = decode(completed["dense"])
    records = []
    review = {"schema_version": 3, "suite": str(suite_path), "status": "PENDING_HUMAN_REVIEW", "families": {}}
    for family, candidates in FAMILIES.items():
        missing = [candidate for candidate in candidates if candidate not in completed]
        if missing:
            raise RuntimeError(f"missing {family} calibration candidates: {missing}")
        rows = []
        for label, video_frames in [("dense", dense)] + [(candidate, decode(completed[candidate])) for candidate in candidates]:
            tiles = []
            for frame_id in (0, 20, 40, 60, 80):
                tile = cv2.cvtColor(video_frames[frame_id], cv2.COLOR_RGB2BGR)
                tile = cv2.resize(tile, (256, 148), interpolation=cv2.INTER_AREA)
                cv2.putText(tile, f"{label} f{frame_id}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(tile)
            rows.append(np.hstack(tiles))
        output = output_dir / f"{family}_contact_sheet.png"
        cv2.imwrite(str(output), np.vstack(rows))
        records.append({"family": family, "candidates": candidates, "contact_sheet": str(output)})
        review["families"][family] = {
            "candidates": {
                candidate: {
                    "visual_status": "pending",
                    "subject_preserved": None,
                    "large_white_or_missing_regions": None,
                    "motion_stable": None,
                    "notes": "",
                }
                for candidate in candidates
            },
            "selected_candidate": None,
            "selection_notes": "",
        }
    (output_dir / "index.json").write_text(json.dumps({"records": records}, indent=2, sort_keys=True) + "\n")
    review_path = ROOT / "configs/stage3_calibration_human_review.json"
    if review_path.exists():
        raise RuntimeError(f"refusing to overwrite existing review: {review_path}")
    review_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"contact_sheets": len(records), "review": str(review_path), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
