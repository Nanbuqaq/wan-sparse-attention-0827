#!/usr/bin/env python3
"""Paired decoded-video metrics and runtime aggregation for a frozen suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import av
import cv2
import numpy as np
import pandas as pd
import torch

from adapters.dependencies import canonical_json, generation_fingerprint, task_fingerprint
from run_matrix import expand_tasks, resolve_common


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_video(path: Path) -> np.ndarray:
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames).astype(np.float32) / 255.0


def psnr(first: np.ndarray, second: np.ndarray) -> float:
    mse = float(np.mean((first - second) ** 2))
    return 100.0 if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def ssim(first: np.ndarray, second: np.ndarray) -> float:
    c1, c2 = 0.01**2, 0.03**2
    values = []
    for channel in range(3):
        x, y = first[..., channel], second[..., channel]
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
        sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
        score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        )
        values.append(float(np.mean(score)))
    return float(np.mean(values))


def optical_flow(video: np.ndarray) -> list[np.ndarray]:
    flows = []
    previous = None
    for frame in video:
        gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (208, 120), interpolation=cv2.INTER_AREA)
        if previous is not None:
            flows.append(
                cv2.calcOpticalFlowFarneback(previous, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            )
        previous = gray
    return flows


def lpips_values(model, dense: np.ndarray, candidate: np.ndarray, device: str) -> list[float]:
    values = []
    for start in range(0, len(dense), 8):
        first = torch.from_numpy(dense[start : start + 8]).permute(0, 3, 1, 2).to(device)
        second = torch.from_numpy(candidate[start : start + 8]).permute(0, 3, 1, 2).to(device)
        first = first * 2.0 - 1.0
        second = second * 2.0 - 1.0
        with torch.inference_mode():
            batch = model(first, second).flatten().detach().cpu().tolist()
        values.extend(float(item) for item in batch)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    suite_path = Path(args.suite).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite["common"] = resolve_common(suite["common"])
    suite_name = suite_path.stem.replace(".template", "")
    output_dir = ROOT / "results" / "metrics" / suite_name
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    inventory_errors = []
    missing_records = []
    expected_tasks = expand_tasks(suite)
    expected_suite_sha = sha256(suite_path)
    for task in expected_tasks:
        video_path = ROOT / task["output"]
        stats_path = video_path.with_suffix(".stats.json")
        error_path = video_path.with_suffix(".error.json")
        if not video_path.is_file() or not stats_path.is_file():
            missing_records.append(
                {
                    "task_id": task["id"],
                    "video": str(video_path),
                    "stats": str(stats_path),
                    "error_file": str(error_path) if error_path.is_file() else None,
                }
            )
            continue
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            inventory_errors.append(f"not completed: {stats_path}")
            continue
        if payload.get("suite_sha256") != expected_suite_sha:
            inventory_errors.append(f"suite hash mismatch: {stats_path}")
            continue
        expected_task_fingerprint = task_fingerprint(task, suite["common"])
        if payload.get("task_fingerprint") != expected_task_fingerprint:
            inventory_errors.append(f"task fingerprint mismatch: {stats_path}")
            continue
        if Path(payload.get("output", "")).resolve() != video_path.resolve():
            inventory_errors.append(f"output path mismatch: {stats_path}")
            continue
        key = canonical_json(generation_fingerprint(task, suite["common"]))
        records.append(
            {
                "stats_path": stats_path,
                "video_path": video_path,
                "payload": payload,
                "task": task,
                "key": key,
            }
        )
    dense = {record["key"]: record for record in records if record["task"]["mode"] == "dense"}
    candidates = [record for record in records if record["task"]["mode"] != "dense"]

    lpips_model = None
    if not args.skip_lpips:
        import lpips

        lpips_model = lpips.LPIPS(net="alex").to(args.device).eval()

    frame_rows = []
    case_rows = []
    decode_errors = []
    dense_cache: dict[str, tuple[np.ndarray, list[np.ndarray]]] = {}
    for record in candidates:
        reference_record = dense.get(record["key"])
        if reference_record is None:
            decode_errors.append({"candidate": str(record["video_path"]), "error": "missing dense reference"})
            continue
        try:
            if record["key"] not in dense_cache:
                dense_video = read_video(reference_record["video_path"])
                dense_cache[record["key"]] = (dense_video, optical_flow(dense_video))
            dense_video, dense_flow = dense_cache[record["key"]]
            candidate = read_video(record["video_path"])
            if candidate.shape != dense_video.shape:
                raise ValueError(f"shape {candidate.shape} != dense {dense_video.shape}")
            flow = optical_flow(candidate)
            lpips_frame = (
                lpips_values(lpips_model, dense_video, candidate, args.device)
                if lpips_model is not None
                else [float("nan")] * len(candidate)
            )
            local_rows = []
            for index, (first, second) in enumerate(zip(dense_video, candidate)):
                row = {
                    "matrix_id": record["task"]["matrix_id"],
                    "prompt_id": record["task"]["prompt_id"],
                    "seed": record["task"]["seed"],
                    "method_id": record["task"]["id"],
                    "base_method_id": record["task"]["base_method_id"],
                    "result_origin": record["task"].get("result_origin", "stage2_new"),
                    "target_density": record["task"].get("density"),
                    "frame": index,
                    "psnr_db": psnr(first, second),
                    "ssim": ssim(first, second),
                    "lpips": lpips_frame[index],
                    "l1": float(np.mean(np.abs(first - second))),
                    "temporal_flicker": (
                        float(
                            np.mean(
                                np.abs(
                                    (second - candidate[index - 1])
                                    - (first - dense_video[index - 1])
                                )
                            )
                        )
                        if index
                        else float("nan")
                    ),
                    "flow_epe": (
                        float(np.linalg.norm(flow[index - 1] - dense_flow[index - 1], axis=-1).mean())
                        if index
                        else float("nan")
                    ),
                }
                frame_rows.append(row)
                local_rows.append(row)
            local = pd.DataFrame(local_rows)
            sparse = record["payload"].get("sparse") or {}
            timing = sparse.get("timing") or {}
            cluster_p50 = (timing.get("cluster") or {}).get("p50_ms")
            permutation_p50 = (timing.get("permutation") or {}).get("p50_ms")
            selection_p50 = (timing.get("selection") or {}).get("p50_ms")
            planner_p50 = (timing.get("planner") or {}).get("p50_ms")
            case_rows.append(
                {
                    "matrix_id": record["task"]["matrix_id"],
                    "prompt_id": record["task"]["prompt_id"],
                    "seed": record["task"]["seed"],
                    "method_id": record["task"]["id"],
                    "base_method_id": record["task"]["base_method_id"],
                    "result_origin": record["task"].get("result_origin", "stage2_new"),
                    "target_density": record["task"].get("density"),
                    "actual_density": sparse.get("logical_pair_density"),
                    "scheduled_density": sparse.get("scheduled_density_vs_dense"),
                    "padding_ratio": sparse.get("padding_ratio"),
                    "load_imbalance_cv": sparse.get("mean_load_imbalance_cv"),
                    "load_imbalance_max_mean": sparse.get("max_load_imbalance_max_mean"),
                    "psnr_mean": float(local["psnr_db"].mean()),
                    "ssim_mean": float(local["ssim"].mean()),
                    "lpips_mean": float(local["lpips"].mean()),
                    "flow_epe_mean": float(local["flow_epe"].mean()),
                    "temporal_flicker": float(local["temporal_flicker"].mean()),
                    "generation_elapsed_s": record["payload"]["generation_elapsed_s"],
                    "peak_memory_bytes": record["payload"]["peak_memory_allocated_bytes"],
                    "peak_memory_reserved_bytes": record["payload"].get("peak_memory_reserved_bytes"),
                    "cluster_p50_ms": cluster_p50,
                    "permutation_p50_ms": permutation_p50,
                    "selection_p50_ms": selection_p50,
                    "planner_p50_ms": planner_p50,
                    "routing_p50_ms": sum(
                        value or 0.0
                        for value in (cluster_p50, permutation_p50, selection_p50, planner_p50)
                    ),
                    "kernel_p50_ms": (timing.get("kernel") or {}).get("p50_ms"),
                    "kernel_warm_p50_ms": (timing.get("kernel_warm") or {}).get("p50_ms"),
                    "kernel_cold_ms": (timing.get("kernel_cold") or {}).get("p50_ms"),
                    "failed_calls": sparse.get("failed_calls", 0),
                    "fallback_calls": sparse.get("dense_fallback_calls", 0),
                    "frames": int(candidate.shape[0]),
                    "height": int(candidate.shape[1]),
                    "width": int(candidate.shape[2]),
                    "video_sha256": sha256(record["video_path"]),
                    "video_path": str(record["video_path"]),
                    "stats_path": str(record["stats_path"]),
                }
            )
        except Exception as error:
            decode_errors.append({"candidate": str(record["video_path"]), "error": repr(error)})

    frame_table = pd.DataFrame(frame_rows)
    case_table = pd.DataFrame(case_rows)
    frame_table.to_csv(output_dir / "frame_metrics.csv", index=False)
    case_table.to_csv(output_dir / "case_metrics.csv", index=False)
    if not case_table.empty:
        summary = (
            case_table.groupby(
                ["result_origin", "base_method_id", "target_density"], dropna=False
            )
            .agg(
                cases=("method_id", "count"),
                psnr_mean=("psnr_mean", "mean"),
                ssim_mean=("ssim_mean", "mean"),
                lpips_mean=("lpips_mean", "mean"),
                flow_epe_mean=("flow_epe_mean", "mean"),
                temporal_flicker=("temporal_flicker", "mean"),
                actual_density=("actual_density", "mean"),
                scheduled_density=("scheduled_density", "mean"),
                padding_ratio=("padding_ratio", "mean"),
                load_imbalance_cv=("load_imbalance_cv", "mean"),
                generation_elapsed_s=("generation_elapsed_s", "mean"),
                cluster_p50_ms=("cluster_p50_ms", "mean"),
                permutation_p50_ms=("permutation_p50_ms", "mean"),
                selection_p50_ms=("selection_p50_ms", "mean"),
                planner_p50_ms=("planner_p50_ms", "mean"),
                routing_p50_ms=("routing_p50_ms", "mean"),
                kernel_p50_ms=("kernel_p50_ms", "mean"),
                kernel_warm_p50_ms=("kernel_warm_p50_ms", "mean"),
                kernel_cold_ms=("kernel_cold_ms", "mean"),
                failed_calls=("failed_calls", "sum"),
                fallback_calls=("fallback_calls", "sum"),
            )
            .reset_index()
        )
    else:
        summary = pd.DataFrame()
    summary.to_csv(output_dir / "method_density_summary.csv", index=False)
    audit = {
        "suite": str(suite_path),
        "suite_sha256": expected_suite_sha,
        "expected_tasks": len(expected_tasks),
        "stats_records": len(records),
        "dense_records": len(dense),
        "candidate_records": len(candidates),
        "evaluated_cases": len(case_rows),
        "decode_errors": decode_errors,
        "inventory_errors": inventory_errors,
        "missing_records": missing_records,
        "lpips_computed": not args.skip_lpips,
        "status": (
            "pass"
            if not decode_errors
            and not inventory_errors
            and not missing_records
            and len(case_rows) == len(candidates)
            else "fail"
        ),
    }
    (output_dir / "evaluation_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
