#!/usr/bin/env python3
"""Audit the isolated SVG2 selection-rule by Dense-guard 2x2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw

from evaluate_matrix import optical_flow, lpips_values, psnr, read_video, ssim


DENSE_VIDEO = ROOT / "results/videos/stage3_formal_50step/stage3_main_d250/gymnast_ribbon/seed_009001/dense.mp4"
REVIEW_PATH = ROOT / "configs/svg2_debug_human_review.json"
OUTPUT_DIR = ROOT / "results/metrics/svg2_debug_20260904"
FIGURE_DIR = ROOT / "results/figures/svg2_debug_20260904"

CELLS = [
    {
        "id": "exact25_varlen_no_guard",
        "origin": "historical",
        "stats": ROOT / "results/videos/formal_50step/main_d250_seed9001/gymnast_ribbon/seed_009001/svg2_varlen_d250.stats.json",
        "method": "svg2_varlen",
        "backend": "varlen_triton",
        "min_k_ratio": 0.0,
        "top_p": None,
        "dense_calls": 0,
        "sparse_calls": 3000,
    },
    {
        "id": "exact25_varlen_guarded",
        "origin": "new",
        "stats": ROOT / "results/videos/svg2_debug_20260904_50step/svg2_debug_2x2/gymnast_ribbon/seed_009001/exact25_varlen_guarded_d250.stats.json",
        "method": "svg2_varlen",
        "backend": "varlen_triton",
        "min_k_ratio": 0.0,
        "top_p": None,
        "dense_calls": 600,
        "sparse_calls": 2400,
    },
    {
        "id": "top_p_varlen_no_guard",
        "origin": "new",
        "stats": ROOT / "results/videos/svg2_debug_20260904_50step/svg2_debug_2x2/gymnast_ribbon/seed_009001/top_p_varlen_no_guard_d250.stats.json",
        "method": "svg2_official_top_p",
        "backend": "varlen_triton",
        "min_k_ratio": 0.1,
        "top_p": 0.9,
        "dense_calls": 0,
        "sparse_calls": 3000,
    },
    {
        "id": "top_p_varlen_guarded",
        "origin": "historical",
        "stats": ROOT / "results/videos/formal_50step/official_top_p_reference/gymnast_ribbon/seed_009001/svg2_official_transfer_d250.stats.json",
        "method": "svg2_official_top_p",
        "backend": "varlen_triton",
        "min_k_ratio": 0.1,
        "top_p": 0.9,
        "dense_calls": 600,
        "sparse_calls": 2400,
    },
]

SMOKE_CELLS = [
    {
        "id": "exact25_varlen_guarded",
        "stats": ROOT / "results/videos/svg2_debug_20260904_smoke/svg2_debug_smoke/gymnast_ribbon/seed_009001/exact25_varlen_guarded_d250.stats.json",
        "dense_calls": 60,
        "sparse_calls": 240,
    },
    {
        "id": "top_p_varlen_no_guard",
        "stats": ROOT / "results/videos/svg2_debug_20260904_smoke/svg2_debug_smoke/gymnast_ribbon/seed_009001/top_p_varlen_no_guard_d250.stats.json",
        "dense_calls": 0,
        "sparse_calls": 300,
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_without_nan(values: list[float]) -> float:
    return float(np.nanmean(np.asarray(values, dtype=np.float64)))


def core_svg2_path_unchanged() -> bool:
    repo = ROOT / "publish_repo"
    command = [
        "git",
        "-C",
        str(repo),
        "diff",
        "--quiet",
        "HEAD",
        "--",
        "adapters/routing.py",
        "adapters/kernels.py",
        "adapters/kernels_varlen_csr.py",
        "adapters/vendor/svoo_repo/svoo/kernels/triton/permute.py",
    ]
    return subprocess.run(command, check=False).returncode == 0


def build_contact_sheet(videos: list[tuple[str, np.ndarray]], output: Path) -> None:
    frame_ids = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    width, height, label_height = 256, 144, 30
    rows = []
    for label, video in videos:
        row = Image.new("RGB", (width * len(frame_ids), height + label_height), "white")
        ImageDraw.Draw(row).text((5, 4), label, fill="black")
        for column, frame_id in enumerate(frame_ids):
            image = Image.fromarray((video[frame_id] * 255.0).round().astype(np.uint8))
            row.paste(image.resize((width, height)), (column * width, label_height))
        rows.append(row)
    sheet = Image.new("RGB", (width * len(frame_ids), (height + label_height) * len(rows)), "white")
    for index, row in enumerate(rows):
        sheet.paste(row, (0, index * (height + label_height)))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-lpips", action="store_true")
    args = parser.parse_args()

    review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    dense = read_video(DENSE_VIDEO)
    dense_flow = optical_flow(dense)
    lpips_model = None
    if not args.skip_lpips:
        import lpips

        lpips_model = lpips.LPIPS(net="alex").to(args.device).eval()

    rows = []
    checks = {
        "review_complete": review.get("status") == "COMPLETE",
        "core_svg2_selection_permutation_and_materialization_unchanged": core_svg2_path_unchanged(),
        "dense_has_81_frames": dense.shape[0] == 81,
    }
    smoke_rows = []
    for spec in SMOKE_CELLS:
        payload = json.loads(spec["stats"].read_text(encoding="utf-8"))
        sparse = payload["sparse"]
        video_path = Path(payload["output"])
        video = read_video(video_path)
        policy = sparse["execution_policy"]["svg2_dense_guard"]
        smoke_row = {
            "cell_id": spec["id"],
            "frames": int(video.shape[0]),
            "video_sha256": sha256(video_path),
            "logical_density": sparse.get("logical_pair_density"),
            "scheduled_density": sparse.get("scheduled_density_vs_dense"),
            "explicit_dense_calls": sparse.get("explicit_dense_reference_calls"),
            "sparse_calls": sparse.get("sparse_kernel_calls"),
            "matches_expected_calls": policy.get("matches_expected_calls"),
        }
        smoke_rows.append(smoke_row)
        prefix = f"smoke_{spec['id']}"
        checks.update(
            {
                f"{prefix}_status_completed": payload.get("status") == "completed",
                f"{prefix}_sha_matches": smoke_row["video_sha256"] == payload.get("output_sha256"),
                f"{prefix}_81_frames": smoke_row["frames"] == 81,
                f"{prefix}_calls_match": (
                    smoke_row["explicit_dense_calls"] == spec["dense_calls"]
                    and smoke_row["sparse_calls"] == spec["sparse_calls"]
                    and smoke_row["matches_expected_calls"] is True
                ),
                f"{prefix}_no_failure_or_fallback": (
                    sparse.get("failed_calls") == 0
                    and sparse.get("dense_fallback_calls") == 0
                ),
            }
        )
    videos = [("Dense", dense)]
    for spec in CELLS:
        payload = json.loads(spec["stats"].read_text(encoding="utf-8"))
        task = payload["task"]
        sparse = payload["sparse"]
        video_path = Path(payload["output"])
        video = read_video(video_path)
        flow = optical_flow(video)
        lpips_frame = (
            lpips_values(lpips_model, dense, video, args.device)
            if lpips_model is not None
            else [float("nan")] * len(video)
        )
        frame_psnr = [psnr(first, second) for first, second in zip(dense, video)]
        frame_ssim = [ssim(first, second) for first, second in zip(dense, video)]
        frame_flow = [
            float(np.linalg.norm(first - second, axis=-1).mean())
            for first, second in zip(dense_flow, flow)
        ]
        frame_flicker = [
            float(np.mean(np.abs((video[index] - video[index - 1]) - (dense[index] - dense[index - 1]))))
            for index in range(1, len(video))
        ]
        cell_review = review["cells"][spec["id"]]
        row = {
            "cell_id": spec["id"],
            "origin": spec["origin"],
            "selection_policy": "top_p_0p9_min_k_0p1" if spec["top_p"] is not None else "exact_0p25_min_k_0",
            "dense_guard": "guarded" if spec["dense_calls"] else "no_guard",
            "backend": task.get("backend"),
            "actual_density": sparse.get("logical_pair_density"),
            "scheduled_density": sparse.get("scheduled_density_vs_dense"),
            "generation_elapsed_s": payload.get("generation_elapsed_s"),
            "psnr_mean": mean_without_nan(frame_psnr),
            "ssim_mean": mean_without_nan(frame_ssim),
            "lpips_mean": mean_without_nan(lpips_frame),
            "flow_epe_mean": mean_without_nan(frame_flow),
            "temporal_flicker": mean_without_nan(frame_flicker),
            "explicit_dense_calls": sparse.get("explicit_dense_reference_calls"),
            "sparse_calls": sparse.get("sparse_kernel_calls"),
            "failed_calls": sparse.get("failed_calls"),
            "fallback_calls": sparse.get("dense_fallback_calls"),
            "frames": int(video.shape[0]),
            "video_sha256": sha256(video_path),
            "video_path": str(video_path),
            "visual_status": cell_review["visual_status"],
            "subject_preserved": cell_review["subject_preserved"],
        }
        rows.append(row)
        videos.append((spec["id"], video))
        prefix = f"cell_{spec['id']}"
        checks.update(
            {
                f"{prefix}_status_completed": payload.get("status") == "completed",
                f"{prefix}_task_matches": (
                    task.get("method") == spec["method"]
                    and task.get("backend") == spec["backend"]
                    and float(task.get("min_k_ratio", 0.1)) == spec["min_k_ratio"]
                    and task.get("top_p") == spec["top_p"]
                ),
                f"{prefix}_sha_matches": row["video_sha256"] == payload.get("output_sha256"),
                f"{prefix}_81_frames": row["frames"] == 81,
                f"{prefix}_calls_match": (
                    row["explicit_dense_calls"] == spec["dense_calls"]
                    and row["sparse_calls"] == spec["sparse_calls"]
                ),
                f"{prefix}_no_failure_or_fallback": row["failed_calls"] == 0 and row["fallback_calls"] == 0,
            }
        )

    by_id = {row["cell_id"]: row for row in rows}
    exact_guarded_pass = by_id["exact25_varlen_guarded"]["visual_status"] == "pass"
    top_p_no_guard_pass = by_id["top_p_varlen_no_guard"]["visual_status"] == "pass"
    if top_p_no_guard_pass and not exact_guarded_pass:
        decision = "selection_exact_budget_is_primary_failure"
    elif exact_guarded_pass and not top_p_no_guard_pass:
        decision = "dense_guard_is_primary_factor"
    elif exact_guarded_pass and top_p_no_guard_pass:
        decision = "either_top_p_or_guard_is_sufficient"
    else:
        decision = "top_p_and_guard_interaction_required"
    checks["decision_matches_human_review"] = decision == review.get("decision")

    guarded_reference = by_id["top_p_varlen_guarded"]
    eligible = [
        row["cell_id"]
        for row in rows
        if row["origin"] == "new"
        and row["visual_status"] == "pass"
        and row["actual_density"] <= guarded_reference["actual_density"]
        and row["generation_elapsed_s"] <= guarded_reference["generation_elapsed_s"]
    ]
    third_video_required = bool(eligible)
    checks["koi_gate_applied"] = not third_video_required

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (OUTPUT_DIR / "four_cell_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    contact_sheet = FIGURE_DIR / "gymnast_svg2_varlen_2x2_9frame.png"
    build_contact_sheet(videos, contact_sheet)
    audit = {
        "schema_version": 1,
        "status": "pass" if all(checks.values()) else "fail",
        "decision": decision,
        "checks": checks,
        "smoke_cells": smoke_rows,
        "cells": rows,
        "third_video_gate": {
            "eligible_new_cells": eligible,
            "third_video_required": third_video_required,
            "density_ceiling": guarded_reference["actual_density"],
            "runtime_ceiling_s": guarded_reference["generation_elapsed_s"],
        },
        "artifacts": {
            "human_review": str(REVIEW_PATH),
            "metrics_csv": str(OUTPUT_DIR / "four_cell_metrics.csv"),
            "contact_sheet": str(contact_sheet),
        },
    }
    (OUTPUT_DIR / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": audit["status"],
        "decision": decision,
        "eligible_new_cells": eligible,
        "third_video_required": third_video_required,
        "output": str(OUTPUT_DIR / "audit.json"),
    }, indent=2))
    if audit["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
