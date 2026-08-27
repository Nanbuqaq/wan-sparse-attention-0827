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


def lpips_distances(reference: np.ndarray, candidate: np.ndarray) -> tuple[list[float] | None, str | None]:
    try:
        import torch
        import lpips
    except Exception as error:
        return None, f"unavailable: {type(error).__name__}: {error}"
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = lpips.LPIPS(net="alex").eval().to(device)
        values = []
        for start in range(0, len(reference), 8):
            ref = torch.from_numpy(reference[start : start + 8]).permute(0, 3, 1, 2)
            cand = torch.from_numpy(candidate[start : start + 8]).permute(0, 3, 1, 2)
            with torch.inference_mode():
                distance = model(ref.to(device) * 2 - 1, cand.to(device) * 2 - 1)
            values.extend(float(value) for value in distance.flatten().cpu())
        return values, None
    except Exception as error:
        return None, f"failed: {type(error).__name__}: {error}"


def embedding_cosines(
    reference: np.ndarray,
    candidate: np.ndarray,
    model_path: Path,
) -> tuple[list[float] | None, str | None]:
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except Exception as error:
        return None, f"unavailable: {type(error).__name__}: {error}"
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = AutoImageProcessor.from_pretrained(
            str(model_path), local_files_only=True
        )
        model = AutoModel.from_pretrained(str(model_path), local_files_only=True).eval().to(device)

        def encode(frames: np.ndarray) -> "torch.Tensor":
            outputs = []
            for start in range(0, len(frames), 8):
                images = [(frame * 255).astype(np.uint8) for frame in frames[start : start + 8]]
                inputs = processor(images=images, return_tensors="pt")
                inputs = {name: value.to(device) for name, value in inputs.items()}
                with torch.inference_mode():
                    result = model(**inputs)
                if getattr(result, "pooler_output", None) is not None:
                    feature = result.pooler_output
                else:
                    feature = result.last_hidden_state[:, 0]
                outputs.append(torch.nn.functional.normalize(feature.float(), dim=-1).cpu())
            return torch.cat(outputs)

        ref = encode(reference)
        cand = encode(candidate)
        values = (ref * cand).sum(dim=-1).tolist()
        return [float(value) for value in values], None
    except Exception as error:
        return None, f"failed: {type(error).__name__}: {error}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", action="append", required=True, help="name=path")
    parser.add_argument("--output-dir", default="results/metrics/video_quality")
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument(
        "--embedding-model",
        action="append",
        default=[],
        help="metric_name=/local/offline/model/path, for example dino=/models/dinov2",
    )
    args = parser.parse_args()
    reference_path = Path(args.reference).resolve()
    reference = read_video(reference_path)
    reference_flow = optical_flow(reference)
    candidates = {}
    for item in args.candidate:
        name, value = item.split("=", 1)
        candidates[name] = Path(value).resolve()
    embedding_models = {}
    for item in args.embedding_model:
        name, value = item.split("=", 1)
        embedding_models[name] = Path(value).resolve()
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
        lpips_values, lpips_error = (
            (None, "disabled")
            if args.no_lpips
            else lpips_distances(reference, video)
        )
        embedding_values = {}
        embedding_errors = {}
        for metric_name, model_path in embedding_models.items():
            values, error = embedding_cosines(reference, video, model_path)
            embedding_values[metric_name] = values
            embedding_errors[metric_name] = error
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
                "lpips": (
                    lpips_values[index] if lpips_values is not None else float("nan")
                ),
            }
            for metric_name, values in embedding_values.items():
                row[f"{metric_name}_cosine"] = (
                    values[index] if values is not None else float("nan")
                )
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
            "lpips_mean": (
                float(np.mean(lpips_values)) if lpips_values is not None else None
            ),
            "late_quarter_lpips_mean": (
                float(np.mean(lpips_values[late_start:]))
                if lpips_values is not None
                else None
            ),
            "lpips_status": "available" if lpips_values is not None else lpips_error,
        }
        for metric_name, values in embedding_values.items():
            summary[name][f"{metric_name}_cosine_mean"] = (
                float(np.mean(values)) if values is not None else None
            )
            summary[name][f"late_quarter_{metric_name}_cosine_mean"] = (
                float(np.mean(values[late_start:])) if values is not None else None
            )
            summary[name][f"{metric_name}_status"] = (
                "available" if values is not None else embedding_errors[metric_name]
            )
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
            "statistical_unit": "complete video; frame rows are diagnostics and are not independent bootstrap samples",
            "embedding_models": {
                name: str(path) for name, path in embedding_models.items()
            },
        },
    }
    (output_dir / "paired_video_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
