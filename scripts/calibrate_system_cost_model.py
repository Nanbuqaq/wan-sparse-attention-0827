#!/usr/bin/env python3
"""Fit and audit a frozen transfer-cost model from isolated replay results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch


def _load_rows(paths: Iterable[Path]) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    hardware: list[dict] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "pass":
            raise ValueError(f"non-pass transfer benchmark: {path}")
        hardware.append(
            {
                "artifact": str(path),
                "gpu": payload.get("gpu"),
                "compute_capability": payload.get("compute_capability"),
                "torch": payload.get("torch"),
                "cuda": payload.get("cuda"),
            }
        )
        for case_id, record in payload["results"].items():
            required = (
                "transfer_mode",
                "transferred_bytes",
                "h2d_copy_count",
                "h2d_s_median",
                "cpu_gather_s_median",
                "pack_run_count",
                "pack_bytes",
            )
            missing = [name for name in required if name not in record]
            if missing:
                raise KeyError(f"{path}:{case_id} missing fields {missing}")
            rows.append(
                {
                    "artifact": str(path),
                    "case_id": case_id,
                    "transfer_mode": str(record["transfer_mode"]),
                    "copied_bytes": float(record["transferred_bytes"]),
                    "h2d_copy_count": float(record["h2d_copy_count"]),
                    "h2d_s": float(record["h2d_s_median"]),
                    "pack_bytes": float(record["pack_bytes"]),
                    "pack_run_count": float(record["pack_run_count"]),
                    "pack_s": float(record["cpu_gather_s_median"]),
                }
            )
    if not rows:
        raise ValueError("no transfer benchmark rows were loaded")
    identities = {
        (item["gpu"], tuple(item["compute_capability"] or ())) for item in hardware
    }
    if len(identities) != 1:
        raise ValueError("one hardware cost profile cannot mix GPU identities")
    return rows, hardware


def _fit_nonnegative(features: list[list[float]], targets: list[float]) -> list[float]:
    if not features or len(features) != len(targets):
        raise ValueError("fit data must be non-empty and aligned")
    width = len(features[0])
    if width < 1 or any(len(row) != width for row in features):
        raise ValueError("fit feature width must be positive and constant")
    matrix = torch.tensor(features, dtype=torch.float64)
    target = torch.tensor(targets, dtype=torch.float64)
    best_coefficients = torch.zeros(width, dtype=torch.float64)
    best_error = float(torch.sum(target.square()))
    for mask in range(1, 1 << width):
        active = [index for index in range(width) if mask & (1 << index)]
        candidate_matrix = matrix[:, active]
        solution = torch.linalg.lstsq(candidate_matrix, target).solution
        if bool((solution < -1e-15).any()):
            continue
        coefficients = torch.zeros(width, dtype=torch.float64)
        coefficients[active] = solution.clamp_min(0)
        error = float(torch.sum((matrix @ coefficients - target).square()))
        if error < best_error:
            best_error = error
            best_coefficients = coefficients
    return [float(value) for value in best_coefficients]


def _artifact_sha(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item.resolve())):
        digest.update(str(path.resolve()).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _predict_row(row: dict, coefficients: dict[str, float]) -> dict:
    predicted_h2d = (
        row["copied_bytes"] * coefficients["h2d_seconds_per_byte"]
        + row["h2d_copy_count"] * coefficients["copy_launch_seconds"]
    )
    has_pack = row["pack_bytes"] > 0 or row["pack_run_count"] > 0
    predicted_pack = 0.0
    if has_pack:
        predicted_pack = (
            coefficients["pack_fixed_seconds"]
            + row["pack_bytes"] * coefficients["pack_seconds_per_byte"]
            + row["pack_run_count"] * coefficients["pack_run_seconds"]
        )
    measured = row["h2d_s"] + row["pack_s"]
    predicted = predicted_h2d + predicted_pack
    if measured <= 0:
        raise ValueError("measured replay time must be positive")
    return {
        **row,
        "predicted_h2d_s": predicted_h2d,
        "predicted_pack_s": predicted_pack,
        "predicted_total_s": predicted,
        "measured_total_s": measured,
        "absolute_percentage_error": abs(predicted - measured) / measured,
    }


def calibrate(
    calibration_paths: list[Path],
    holdout_paths: list[Path],
    *,
    profile_id: str,
    model_version: str,
    hbm_bytes_per_second: float,
    mape_gate: float = 0.15,
) -> dict:
    if hbm_bytes_per_second <= 0:
        raise ValueError("hbm_bytes_per_second must be positive")
    if not 0 < mape_gate < 1:
        raise ValueError("mape_gate must be in (0,1)")
    calibration_rows, calibration_hardware = _load_rows(calibration_paths)
    holdout_rows, holdout_hardware = _load_rows(holdout_paths)
    hardware_identities = {
        (item["gpu"], tuple(item["compute_capability"] or ()))
        for item in calibration_hardware + holdout_hardware
    }
    if len(hardware_identities) != 1:
        raise ValueError("calibration and held-out replays must use one hardware identity")

    h2d_per_byte, copy_launch = _fit_nonnegative(
        [[row["copied_bytes"], row["h2d_copy_count"]] for row in calibration_rows],
        [row["h2d_s"] for row in calibration_rows],
    )
    packed_rows = [
        row
        for row in calibration_rows
        if row["pack_bytes"] > 0 or row["pack_run_count"] > 0
    ]
    if not packed_rows:
        raise ValueError("calibration requires packed transfer rows")
    pack_per_byte, pack_per_run, pack_fixed = _fit_nonnegative(
        [
            [row["pack_bytes"], row["pack_run_count"], 1.0]
            for row in packed_rows
        ],
        [row["pack_s"] for row in packed_rows],
    )
    coefficients = {
        "h2d_seconds_per_byte": h2d_per_byte,
        "copy_launch_seconds": copy_launch,
        "pack_seconds_per_byte": pack_per_byte,
        "pack_run_seconds": pack_per_run,
        "pack_fixed_seconds": pack_fixed,
    }
    audited_calibration = [_predict_row(row, coefficients) for row in calibration_rows]
    audited_holdout = [_predict_row(row, coefficients) for row in holdout_rows]
    holdout_mape = sum(
        row["absolute_percentage_error"] for row in audited_holdout
    ) / len(audited_holdout)
    source_paths = calibration_paths + holdout_paths
    epsilon = 1e-18
    profile = {
        "profile_id": profile_id,
        "model_version": model_version,
        "h2d_bytes_per_second": 1.0 / max(h2d_per_byte, epsilon),
        "hbm_bytes_per_second": hbm_bytes_per_second,
        "copy_launch_seconds": max(copy_launch, 1e-12),
        "pack_run_seconds": max(pack_per_run, 1e-12),
        "pack_bytes_per_second": 1.0 / max(pack_per_byte, epsilon),
        "pack_fixed_seconds": pack_fixed,
        "source_artifact_sha256": _artifact_sha(source_paths),
    }
    allowed = holdout_mape <= mape_gate
    return {
        "status": "pass" if allowed else "negative",
        "cost_aware_admission_allowed": allowed,
        "heldout_mape": holdout_mape,
        "mape_gate": mape_gate,
        "profile": profile,
        "coefficients": coefficients,
        "hardware": calibration_hardware + holdout_hardware,
        "calibration_rows": audited_calibration,
        "holdout_rows": audited_holdout,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", action="append", required=True)
    parser.add_argument("--holdout", action="append", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--model-version", default="transfer-nnls-v1")
    parser.add_argument("--hbm-bytes-per-second", type=float, required=True)
    parser.add_argument("--mape-gate", type=float, default=0.15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = calibrate(
        [Path(value) for value in args.calibration],
        [Path(value) for value in args.holdout],
        profile_id=args.profile_id,
        model_version=args.model_version,
        hbm_bytes_per_second=args.hbm_bytes_per_second,
        mape_gate=args.mape_gate,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "heldout_mape": payload["heldout_mape"],
                "cost_aware_admission_allowed": payload[
                    "cost_aware_admission_allowed"
                ],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
