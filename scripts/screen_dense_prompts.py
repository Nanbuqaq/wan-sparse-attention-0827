#!/usr/bin/env python3
"""Build Dense-only contact sheets and automatic validity evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()

import av
import cv2
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_video(path: Path) -> np.ndarray:
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        metadata = {
            "width": stream.width,
            "height": stream.height,
            "fps": float(stream.average_rate),
        }
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise RuntimeError(f"no frames decoded: {path}")
    return np.stack(frames), metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dense_prompt_candidates_v2.json")
    parser.add_argument("--scores")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    scores = {}
    if args.scores:
        scores = json.loads((ROOT / args.scores).read_text(encoding="utf-8"))
    figure_dir = ROOT / "results" / "figures" / "dense_prompt_screen_v2"
    figure_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for candidate in config["candidates"]:
        relative = candidate.get("stage1_video")
        if relative is None:
            rows.append({"id": candidate["id"], "status": "needs_dense_generation"})
            continue
        path = ROOT / relative
        try:
            video, metadata = read_video(path)
            selected_ids = np.linspace(0, len(video) - 1, 5).round().astype(int)
            tiles = []
            for frame_id in selected_ids:
                tile = cv2.cvtColor(video[frame_id], cv2.COLOR_RGB2BGR)
                tile = cv2.resize(tile, (320, 185), interpolation=cv2.INTER_AREA)
                cv2.putText(tile, f"{candidate['id']} f{frame_id}", (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(tile)
            contact = np.hstack(tiles)
            contact_path = figure_dir / f"{candidate['id']}.png"
            cv2.imwrite(str(contact_path), contact)
            normalized = video.astype(np.float32) / 255.0
            motion = float(np.mean(np.abs(normalized[1:] - normalized[:-1])))
            variance = float(np.var(normalized))
            human = scores.get(candidate["id"])
            accepted = None
            if human is not None:
                components = [int(human[key]) for key in ("prompt_match", "structure", "artifacts", "motion", "flicker")]
                accepted = sum(components) >= 8 and min(components) > 0
            rows.append(
                {
                    "id": candidate["id"],
                    "status": "decoded",
                    "path": str(path),
                    "sha256": sha256(path),
                    "frames": len(video),
                    **metadata,
                    "pixel_variance": variance,
                    "mean_frame_motion_l1": motion,
                    "contact_sheet": str(contact_path),
                    "human_scores": human,
                    "accepted": accepted,
                }
            )
        except Exception as error:
            rows.append({"id": candidate["id"], "status": "failed", "error": repr(error)})
    accepted_rows = [row for row in rows if row.get("accepted")]
    payload = {
        "schema_version": 2,
        "config": args.config,
        "rows": rows,
        "accepted": [row["id"] for row in accepted_rows],
        "status": "pass" if not args.scores or len(accepted_rows) >= 4 else "needs_more_dense",
    }
    output = ROOT / "results" / "metrics" / "dense_prompt_screen_v2.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "accepted": payload["accepted"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()

