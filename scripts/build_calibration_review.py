#!/usr/bin/env python3
"""Create Dense/candidate contact sheets for 50-step parameter review."""

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


def frames(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        values = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    if len(values) != 81:
        raise RuntimeError(f"expected 81 frames: {path}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    args = parser.parse_args()
    suite_path = Path(args.suite).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite["common"] = resolve_common(suite["common"])
    tasks = expand_tasks(suite)
    dense = {}
    sparse = {}
    for task in tasks:
        video = ROOT / task["output"]
        stats = video.with_suffix(".stats.json")
        if not video.is_file() or not stats.is_file():
            continue
        if task["mode"] == "dense":
            dense[(task["prompt_id"], task["seed"])] = video
        else:
            method, candidate = task["base_method_id"].split("__", 1)
            sparse[(method, candidate, task["prompt_id"], task["seed"], task["density"])] = video
    output_dir = ROOT / "results/figures/calibration_review_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    methods = sorted({key[0] for key in sparse})
    for method in methods:
        candidates = sorted({key[1] for key in sparse if key[0] == method})
        if len(candidates) != 2:
            raise RuntimeError(f"expected two candidates for {method}: {candidates}")
        for prompt_id in sorted({key[2] for key in sparse if key[0] == method}):
            for density in sorted({key[4] for key in sparse if key[0] == method and key[2] == prompt_id}):
                paths = [
                    ("dense", dense[(prompt_id, 42)]),
                    (candidates[0], sparse[(method, candidates[0], prompt_id, 42, density)]),
                    (candidates[1], sparse[(method, candidates[1], prompt_id, 42, density)]),
                ]
                rows = []
                for label, path in paths:
                    video_frames = frames(path)
                    tiles = []
                    for frame_id in (0, 40, 80):
                        tile = cv2.cvtColor(video_frames[frame_id], cv2.COLOR_RGB2BGR)
                        tile = cv2.resize(tile, (320, 185), interpolation=cv2.INTER_AREA)
                        cv2.putText(tile, f"{label} f{frame_id}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
                        tiles.append(tile)
                    rows.append(np.hstack(tiles))
                output = output_dir / f"{method}__{prompt_id}__d{int(density * 1000):03d}.png"
                cv2.imwrite(str(output), np.vstack(rows))
                records.append({"method": method, "prompt_id": prompt_id, "density": density, "candidates": candidates, "contact_sheet": str(output)})
    index = output_dir / "index.json"
    index.write_text(json.dumps({"records": records}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"contact_sheets": len(records), "index": str(index)}, indent=2))


if __name__ == "__main__":
    main()

