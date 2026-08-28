#!/usr/bin/env python3
"""Audit the complete method-smoke stage, including auxiliary GPU gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import av
import torch


PAPER_METHODS = {"svg2_ar", "adacluster_ar", "svoo_ar", "scope_ar"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decoded_frames(path: Path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def audit_cases(
    expected: list[dict],
    state_payloads: list[dict],
    *,
    verify_artifacts: bool,
) -> tuple[list[dict], list[str]]:
    merged: dict[str, dict] = {}
    errors: list[str] = []
    for payload in state_payloads:
        for case in payload["cases"]:
            case_id = case.get("id", case.get("case_id"))
            if not case_id:
                errors.append("case without id")
                continue
            if case_id in merged and merged[case_id] != case:
                errors.append(f"conflicting duplicate case: {case_id}")
                continue
            merged[case_id] = case

    expected_ids = {case["id"] for case in expected}
    missing = sorted(expected_ids - set(merged))
    extra = sorted(set(merged) - expected_ids)
    errors.extend(f"missing case: {case_id}" for case_id in missing)
    errors.extend(f"unexpected case: {case_id}" for case_id in extra)

    records = []
    for case_id in sorted(expected_ids & set(merged)):
        case = merged[case_id]
        records.append(case)
        if case.get("status") != "pass":
            errors.append(
                f"method smoke is not pass: {case_id}: {case.get('status')} "
                f"{case.get('failure_reason', '')}"
            )
            continue
        for key in ("backend", "stats", "config", "route_plan_sha256"):
            if not case.get(key):
                errors.append(f"pass case missing {key}: {case_id}")
        if any(case.get(key, 0) for key in ("failed_calls", "fallback_calls", "nan_calls")):
            errors.append(f"pass case has failed/fallback/NaN calls: {case_id}")
        if not verify_artifacts:
            continue
        video = Path(str(case.get("video", "")))
        latent = video.with_name("latents.pt")
        if not video.is_file() or not latent.is_file():
            errors.append(f"pass case missing video/latent artifact: {case_id}")
            continue
        decoded = _decoded_frames(video)
        if decoded != 81:
            errors.append(f"decoded frames {decoded} != 81: {case_id}")
        if case.get("video_sha256") and _sha256(video) != case["video_sha256"]:
            errors.append(f"video SHA mismatch: {case_id}")
        tensor = torch.load(latent, map_location="cpu", weights_only=True)
        if tuple(tensor.shape) != (1, 21, 16, 60, 104):
            errors.append(f"latent shape mismatch {tuple(tensor.shape)}: {case_id}")
        if not torch.isfinite(tensor).all():
            errors.append(f"non-finite latent: {case_id}")
    return records, errors


def _audit_auxiliary(args: argparse.Namespace, errors: list[str]) -> dict:
    auxiliary: dict = {}
    for name, directory, frames, latent_frames in (
        ("rag_dense_39", Path(args.rag39_dir), 153, 39),
        ("block64_100pct_21", Path(args.block100_dir), 81, 21),
    ):
        videos = sorted(directory.glob("*.mp4"))
        video = videos[0] if videos else directory / "missing.mp4"
        latent = directory / "latents.pt"
        stats_path = directory / "sparse_history_stats.json"
        record = {"video": str(video), "latent": str(latent), "stats": str(stats_path)}
        auxiliary[name] = record
        if not (video.is_file() and latent.is_file() and stats_path.is_file()):
            errors.append(f"missing auxiliary artifacts: {name}")
            continue
        record["decoded_frames"] = _decoded_frames(video)
        if record["decoded_frames"] != frames:
            errors.append(
                f"decoded frames {record['decoded_frames']} != {frames}: {name}"
            )
        tensor = torch.load(latent, map_location="cpu", weights_only=True)
        record["latent_shape"] = list(tensor.shape)
        record["latent_finite"] = bool(torch.isfinite(tensor).all())
        if tuple(tensor.shape) != (1, latent_frames, 16, 60, 104):
            errors.append(f"latent shape mismatch {tuple(tensor.shape)}: {name}")
        if not record["latent_finite"]:
            errors.append(f"non-finite latent: {name}")
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        record["failed_calls"] = stats.get("failed_calls", 0)
        record["fallback_calls"] = stats.get("dense_fallback_calls", 0)
        if record["failed_calls"] or record["fallback_calls"]:
            errors.append(f"auxiliary stats contain failures/fallbacks: {name}")

    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    calibration_methods = set(calibration.get("method_params", {}))
    auxiliary["calibration"] = {
        "status": calibration.get("status"),
        "methods": sorted(calibration_methods),
        "formal_prompts_used": calibration.get("formal_prompts_used"),
    }
    if (
        calibration.get("status") != "frozen_before_method_smoke"
        or calibration_methods != PAPER_METHODS
        or calibration.get("formal_prompts_used") is not False
    ):
        errors.append("paper calibration is incomplete or used formal prompts")

    equivalence = json.loads(Path(args.latent_equivalence).read_text(encoding="utf-8"))
    auxiliary["latent_equivalence"] = equivalence
    if (
        equivalence.get("status") != "pass"
        or float(equivalence.get("relative_l2", float("inf"))) > 0.01
    ):
        errors.append("matched 100% latent equivalence gate failed")

    benchmark_payload = json.loads(
        Path(args.backend_benchmark).read_text(encoding="utf-8")
    )
    benchmarks = benchmark_payload.get("benchmarks", [benchmark_payload])
    auxiliary["backend_benchmark"] = benchmark_payload
    if benchmark_payload.get("benchmarks") is not None and len(benchmarks) != 4:
        errors.append("four-density backend benchmark batch is incomplete")
    densities = {round(float(item.get("density", -1)), 2) for item in benchmarks}
    if densities != {0.10, 0.15, 0.25, 1.00}:
        errors.append(f"backend benchmark densities are incomplete: {sorted(densities)}")
    kernel_negatives = []
    for benchmark in benchmarks:
        backend_records = benchmark.get("backends", {})
        if set(backend_records) != {"grouped_fa2", "fixed64_rect", "varlen_triton"}:
            errors.append("backend benchmark is incomplete")
            continue
        route_shas = {
            record.get("route_plan_sha256") for record in backend_records.values()
        }
        if len(route_shas) != 1 or None in route_shas:
            errors.append(
                f"backend route-plan SHA differs at density {benchmark.get('density')}"
            )
        for backend, record in backend_records.items():
            if not record.get("same_route_plan"):
                errors.append(f"backend route-plan mismatch: {backend}")
            if record.get("warmup") != 5 or record.get("iterations") != 20:
                errors.append(f"backend benchmark is not warm 5+20: {backend}")
            if record.get("backend_ms_median") is None or record.get("wall_ms_median") is None:
                errors.append(f"backend benchmark missing timing: {backend}")
            if backend == "grouped_fa2" and record.get("status") != "pass":
                errors.append(
                    f"required grouped FA2 numerical gate failed at density {benchmark.get('density')}"
                )
            elif backend != "grouped_fa2" and record.get("status") != "pass":
                metrics = record.get("error_vs_grouped", {})
                if not {"max_abs", "relative_l2", "cosine"}.issubset(metrics):
                    errors.append(f"kernel negative missing numerical evidence: {backend}")
                kernel_negatives.append(
                    {
                        "density": benchmark.get("density"),
                        "backend": backend,
                        "status": "negative",
                        "reason": "frozen different-kernel numerical threshold exceeded",
                        "error_vs_grouped": metrics,
                    }
                )
    auxiliary["backend_benchmark_interpretation"] = {
        "required_quality_backend": "grouped_fa2",
        "optional_kernel_failures_are_performance_negatives": True,
        "kernel_negatives": kernel_negatives,
    }
    return auxiliary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", default="configs/rag_smoke_expected.json")
    parser.add_argument("--states", action="append", required=True)
    parser.add_argument("--rag39-dir", required=True)
    parser.add_argument("--block100-dir", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--latent-equivalence", required=True)
    parser.add_argument("--backend-benchmark", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))["cases"]
    state_payloads = [
        json.loads(Path(path).read_text(encoding="utf-8")) for path in args.states
    ]
    records, errors = audit_cases(expected, state_payloads, verify_artifacts=True)
    auxiliary = _audit_auxiliary(args, errors)
    payload = {
        "status": "pass" if not errors else "fail",
        "commit": args.commit,
        "expected_cases": len(expected),
        "pass_cases": sum(case.get("status") == "pass" for case in records),
        "errors": errors,
        "cases": records,
        "auxiliary": auxiliary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
