#!/usr/bin/env python3
"""Create multi-frame formal contact sheets and a Stage-3 review template."""

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


METHODS = ["dense", "block", "scope", "svg2", "coverage_cluster", "vaware_cluster", "stage3_hybrid"]
CASES = [
    ("gymnast_ribbon", 9001, "main_gymnast"),
    ("skateboard_alley", 9001, "main_skateboard"),
    ("koi_reflections", 9001, "main_koi"),
    ("orchestra_conductor", 9001, "main_orchestra"),
    ("gymnast_ribbon", 65537, "second_seed_gymnast"),
    ("fox_snow", 9001, "negative_fox"),
    ("glassblower", 9001, "negative_glassblower"),
]


def decode(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    if len(frames) != 81:
        raise RuntimeError(f"expected 81 frames: {path}")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="configs/stage3_formal_50step.json")
    args = parser.parse_args()
    suite_path = ROOT / args.suite
    suite = json.loads(suite_path.read_text())
    suite["common"] = resolve_common(suite["common"])
    lookup = {}
    for task in expand_tasks(suite):
        video = ROOT / task["output"]
        stats = video.with_suffix(".stats.json")
        if video.is_file() and stats.is_file() and json.loads(stats.read_text()).get("status") == "completed":
            lookup[(task["prompt_id"], int(task["seed"]), task["base_method_id"])] = video
    output_dir = ROOT / "results/figures/stage3_formal_review"
    output_dir.mkdir(parents=True, exist_ok=True)
    review = {"schema_version": 3, "status": "PENDING_HUMAN_REVIEW", "suite": args.suite, "cases": {}}
    records = []
    for prompt, seed, name in CASES:
        rows = []
        for method in METHODS:
            video = lookup[(prompt, seed, method)]
            frames = decode(video)
            tiles = []
            for frame_id in (0, 20, 40, 60, 80):
                tile = cv2.cvtColor(frames[frame_id], cv2.COLOR_RGB2BGR)
                tile = cv2.resize(tile, (256, 148), interpolation=cv2.INTER_AREA)
                cv2.putText(tile, f"{method} f{frame_id}", (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(tile)
            rows.append(np.hstack(tiles))
        output = output_dir / f"{name}.png"
        cv2.imwrite(str(output), np.vstack(rows))
        key = f"{prompt}__seed{seed}"
        review["cases"][key] = {
            "prompt_id": prompt,
            "seed": seed,
            "contact_sheet": str(output.relative_to(ROOT)),
            "case_type": "main" if name.startswith("main") else "second_seed" if name.startswith("second") else "negative",
            "methods": {
                method: {
                    "visual_status": "pending",
                    "subject_preserved": None,
                    "large_white_or_missing_regions": None,
                    "temporal_stability": "pending",
                    "notes": "",
                }
                for method in METHODS
            },
        }
        records.append({"case": key, "contact_sheet": str(output.relative_to(ROOT))})
    (output_dir / "index.json").write_text(json.dumps({"records": records}, indent=2, sort_keys=True) + "\n")
    review_path = ROOT / "configs/stage3_formal_human_review.json"
    if review_path.exists():
        raise RuntimeError(f"refusing to overwrite {review_path}")
    review_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"contact_sheets": len(records), "review": str(review_path)}, indent=2))


if __name__ == "__main__":
    main()
