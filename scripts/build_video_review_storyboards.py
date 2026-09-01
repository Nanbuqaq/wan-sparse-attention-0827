#!/usr/bin/env python3
"""Build full-video diagnostics and quarter storyboards from case states."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


REVIEWABLE_STATUSES = {"pass", "negative"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_decoded_frames(case: dict) -> int:
    latent_frames = int(case["latent_frames"])
    if latent_frames <= 0:
        raise ValueError(f"invalid latent frame count: {latent_frames}")
    return 4 * latent_frames - 3


def runs(mask: np.ndarray, minimum: int) -> list[tuple[int, int]]:
    output = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(mask) - 1):
            end = index if value and index == len(mask) - 1 else index - 1
            if end - start + 1 >= minimum:
                output.append((start, end))
            start = None
    return output


def quarter_sample_indices(frame_count: int, samples: int) -> list[np.ndarray]:
    if frame_count <= 0 or samples <= 0:
        raise ValueError("frame_count and samples must be positive")
    boundaries = np.linspace(0, frame_count, 5).round().astype(int)
    output = []
    for quarter in range(4):
        start = int(boundaries[quarter])
        end = int(boundaries[quarter + 1]) - 1
        output.append(np.linspace(start, end, samples).round().astype(int))
    return output


def storyboard(
    frames: list[np.ndarray],
    indices: np.ndarray,
    output: Path,
    *,
    columns: int = 8,
) -> None:
    width, height = 208, 120
    rows = math.ceil(len(indices) / columns)
    canvas = Image.new("RGB", (columns * width, rows * height), "black")
    for slot, index in enumerate(indices):
        image = Image.fromarray(frames[int(index)])
        canvas.paste(image, ((slot % columns) * width, (slot // columns) * height))
    canvas.save(output)


def decode_frames(video: Path) -> list[np.ndarray]:
    import av

    frames = []
    with av.open(str(video)) as container:
        for frame in container.decode(video=0):
            frames.append(
                frame.reformat(width=208, height=120).to_ndarray(format="rgb24")
            )
    return frames


def analyze(case: dict, output_root: Path, *, samples_per_quarter: int) -> dict:
    case_id = str(case.get("id", case.get("case_id")))
    video = Path(str(case["video"]))
    if not video.is_file():
        raise FileNotFoundError(f"{case_id}: video missing: {video}")
    video_sha = sha256(video)
    recorded_sha = case.get("video_sha256")
    if recorded_sha and recorded_sha != video_sha:
        raise RuntimeError(f"{case_id}: video SHA mismatch")

    frames = decode_frames(video)
    expected_frames = expected_decoded_frames(case)
    if len(frames) != expected_frames:
        raise RuntimeError(
            f"{case_id}: decoded {len(frames)} frames instead of {expected_frames}"
        )
    gray_array = np.stack(
        [frame.astype(np.float32).mean(axis=2) / 255.0 for frame in frames]
    )
    frame_diff = np.abs(np.diff(gray_array, axis=0)).mean(axis=(1, 2))
    median = float(np.median(frame_diff))
    mad = float(np.median(np.abs(frame_diff - median)))
    freeze_threshold = max(0.0015, 0.08 * median)
    freeze_runs = runs(frame_diff <= freeze_threshold, minimum=8)
    cut_threshold = max(0.12, median + 8.0 * max(mad, 1e-6))
    cut_indices = np.flatnonzero(frame_diff >= cut_threshold).tolist()
    brightness = gray_array.mean(axis=(1, 2))
    brightness_accel = np.abs(np.diff(brightness, n=2))
    flicker_median = float(np.median(brightness_accel))
    flicker_mad = float(np.median(np.abs(brightness_accel - flicker_median)))
    flicker_threshold = max(
        0.025, flicker_median + 8.0 * max(flicker_mad, 1e-6)
    )
    flicker_indices = np.flatnonzero(
        brightness_accel >= flicker_threshold
    ).tolist()

    output_root.mkdir(parents=True, exist_ok=True)
    storyboards = []
    overview = output_root / f"{case_id}__all.png"
    storyboard(
        frames,
        np.linspace(0, len(frames) - 1, samples_per_quarter).round().astype(int),
        overview,
    )
    for quarter, indices in enumerate(
        quarter_sample_indices(len(frames), samples_per_quarter), start=1
    ):
        path = output_root / f"{case_id}__q{quarter}.png"
        storyboard(frames, indices, path)
        storyboards.append(str(path))

    quarter = len(frame_diff) // 4
    early_motion = float(frame_diff[:quarter].mean())
    late_motion = float(frame_diff[-quarter:].mean())
    return {
        "case_id": case_id,
        "case_key_sha256": case.get("case_key_sha256"),
        "commit": case.get("commit"),
        "method": case.get("method", case.get("runtime")),
        "routing_stage": case.get("routing_stage"),
        "prompt_id": case.get("prompt_id"),
        "seed": case.get("seed"),
        "latent_frames": int(case["latent_frames"]),
        "status": case.get("status"),
        "video": str(video),
        "video_sha256": video_sha,
        "decoded_frames": len(frames),
        "expected_decoded_frames": expected_frames,
        "frame_diff_mean": float(frame_diff.mean()),
        "frame_diff_median": median,
        "frame_diff_p95": float(np.quantile(frame_diff, 0.95)),
        "freeze_threshold": freeze_threshold,
        "freeze_runs": freeze_runs,
        "cut_threshold": cut_threshold,
        "cut_indices": cut_indices,
        "flicker_threshold": flicker_threshold,
        "flicker_indices": flicker_indices,
        "early_motion_mean": early_motion,
        "late_motion_mean": late_motion,
        "late_to_early_motion_ratio": late_motion / max(early_motion, 1e-8),
        "overview": str(overview),
        "storyboards": storyboards,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--samples-per-quarter", type=int, default=32)
    args = parser.parse_args()
    if args.samples_per_quarter <= 0:
        parser.error("--samples-per-quarter must be positive")

    output_root = Path(args.output_root)
    records = []
    source_states = []
    seen = set()
    for state_file_value in args.states:
        state_file = Path(state_file_value)
        source_states.append(
            {"path": str(state_file.resolve()), "sha256": sha256(state_file)}
        )
        payload = json.loads(state_file.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            if case.get("status") not in REVIEWABLE_STATUSES:
                continue
            case_id = str(case.get("id", case.get("case_id")))
            if case_id in seen:
                raise RuntimeError(f"duplicate review case: {case_id}")
            seen.add(case_id)
            record_path = output_root / f"{case_id}.json"
            if record_path.is_file():
                existing = json.loads(record_path.read_text(encoding="utf-8"))
                expected = expected_decoded_frames(case)
                required_images = [existing.get("overview"), *existing.get("storyboards", [])]
                if (
                    existing.get("video_sha256") == case.get("video_sha256")
                    and existing.get("expected_decoded_frames") == expected
                    and required_images
                    and all(Path(str(path)).is_file() for path in required_images)
                ):
                    records.append(existing)
                    continue
            record = analyze(
                case,
                output_root,
                samples_per_quarter=args.samples_per_quarter,
            )
            record_path.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            records.append(record)
    records.sort(key=lambda item: (str(item["method"]), str(item["prompt_id"]), int(item["seed"])))
    manifest = {
        "status": "pass",
        "cases": records,
        "source_states": source_states,
        "review_rule": {
            "full_decode": True,
            "expected_frames": "4 * latent_frames - 3",
            "quarter_storyboards": 4,
            "samples_per_quarter": args.samples_per_quarter,
            "statistical_unit": "complete video",
        },
    }
    (output_root / "diagnostics_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"cases": len(records), "output": str(output_root)}, indent=2))


if __name__ == "__main__":
    main()
