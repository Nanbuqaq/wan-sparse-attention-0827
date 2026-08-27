#!/usr/bin/env python3
"""Paired long-video fidelity and late-horizon metrics against a dense reference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import av
import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_video(path: Path) -> np.ndarray:
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames).astype(np.float32) / 255.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def psnr(first: np.ndarray, second: np.ndarray) -> float:
    mse = float(np.mean((first - second) ** 2))
    return 100.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def ssim(first: np.ndarray, second: np.ndarray) -> float:
    c1, c2 = 0.01**2, 0.03**2
    values = []
    for channel in range(3):
        x, y = first[..., channel], second[..., channel]
        mux = cv2.GaussianBlur(x, (11, 11), 1.5)
        muy = cv2.GaussianBlur(y, (11, 11), 1.5)
        varx = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mux * mux
        vary = cv2.GaussianBlur(y * y, (11, 11), 1.5) - muy * muy
        cov = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mux * muy
        values.append(float(np.mean(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (varx + vary + c2)))))
    return float(np.mean(values))


def optical_flow(video: np.ndarray) -> list[np.ndarray]:
    output = []
    previous = None
    for frame in video:
        gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (208, 120), interpolation=cv2.INTER_AREA)
        if previous is not None:
            output.append(cv2.calcOpticalFlowFarneback(previous, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0))
        previous = gray
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", action="append", required=True, help="name=path")
    parser.add_argument("--output-dir", default="results/metrics/video_quality")
    args = parser.parse_args()
    reference_path = Path(args.reference).resolve()
    reference = read_video(reference_path)
    reference_flow = optical_flow(reference)
    candidates = {}
    for item in args.candidate:
        name, value = item.split("=", 1)
        candidates[name] = Path(value).resolve()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_rows = []
    summary = {}
    late_start = (3 * len(reference)) // 4
    for name, path in candidates.items():
        video = read_video(path)
        if video.shape != reference.shape:
            raise ValueError(f"{name} shape {video.shape} != reference {reference.shape}")
        flow = optical_flow(video)
        method_rows = []
        for index, (dense_frame, candidate_frame) in enumerate(zip(reference, video)):
            row = {
                "method": name,
                "frame": index,
                "psnr_db": psnr(dense_frame, candidate_frame),
                "ssim": ssim(dense_frame, candidate_frame),
                "l1": float(np.mean(np.abs(dense_frame - candidate_frame))),
                "temporal_delta_l1": (
                    float(np.mean(np.abs((candidate_frame - video[index - 1]) - (dense_frame - reference[index - 1]))))
                    if index else float("nan")
                ),
                "flow_epe": (
                    float(np.linalg.norm(flow[index - 1] - reference_flow[index - 1], axis=-1).mean())
                    if index else float("nan")
                ),
            }
            frame_rows.append(row)
            method_rows.append(row)
        late = method_rows[late_start:]
        summary[name] = {
            "path": str(path),
            "sha256": sha256(path),
            "decoded_frames": len(video),
            "psnr_mean": float(np.mean([row["psnr_db"] for row in method_rows])),
            "ssim_mean": float(np.mean([row["ssim"] for row in method_rows])),
            "late_quarter_psnr_mean": float(np.mean([row["psnr_db"] for row in late])),
            "late_quarter_ssim_mean": float(np.mean([row["ssim"] for row in late])),
            "temporal_delta_l1_mean": float(np.nanmean([row["temporal_delta_l1"] for row in method_rows])),
            "flow_epe_mean": float(np.nanmean([row["flow_epe"] for row in method_rows])),
        }
    with (output_dir / "paired_frame_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)
    payload = {
        "reference": {
            "path": str(reference_path),
            "sha256": sha256(reference_path),
            "shape": list(reference.shape),
        },
        "candidates": summary,
        "notes": {
            "paired_metrics": "same prompt, seed, sampler, machine, and decoded frame alignment",
            "late_quarter": "last 25 percent of decoded frames",
            "manual_review": "identity, irreversible state reset, action reset, flicker, and freeze remain manually audited",
        },
    }
    (output_dir / "paired_video_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

