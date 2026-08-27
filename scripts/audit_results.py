#!/usr/bin/env python3
"""Audit suite completeness, densities, fallback, decoding, and hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import av

from adapters.dependencies import task_fingerprint
from run_matrix import expand_tasks, resolve_common


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def decoded_frames(path: Path) -> int:
    count = 0
    with av.open(str(path)) as container:
        for _ in container.decode(video=0):
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--density-tolerance", type=float, default=5e-4)
    args = parser.parse_args()
    suite_path = Path(args.suite).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite["common"] = resolve_common(suite["common"])
    tasks = expand_tasks(suite)
    missing, failed, errors = [], [], []
    completed = []
    origins: dict[str, int] = {}
    max_density_error = 0.0
    fallback_calls = 0
    failed_calls = 0
    for task in tasks:
        output = ROOT / task["output"]
        stats_path = output.with_suffix(".stats.json")
        error_path = output.with_suffix(".error.json")
        if error_path.is_file():
            failed.append(str(error_path))
        if not output.is_file() or not stats_path.is_file():
            missing.append(str(output))
            continue
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            errors.append(f"not completed: {stats_path}")
            continue
        expected_suite_sha = sha256(suite_path)
        if payload.get("suite_sha256") != expected_suite_sha:
            errors.append(f"suite sha mismatch: {stats_path}")
        expected_task_fingerprint = task_fingerprint(task, suite["common"])
        if payload.get("task_fingerprint") != expected_task_fingerprint:
            errors.append(f"task fingerprint mismatch: {stats_path}")
        dependency_manifest = payload.get("execution_dependency_manifest") or {}
        if not dependency_manifest.get("task_execution_hash"):
            errors.append(f"missing execution dependency manifest: {stats_path}")
        if Path(payload.get("output", "")).resolve() != output.resolve():
            errors.append(f"output path mismatch: {stats_path}")
        expected_frames = int(task.get("frames", suite["common"]["frames"]))
        actual_frames = decoded_frames(output)
        if actual_frames != expected_frames:
            errors.append(f"frame count {actual_frames} != {expected_frames}: {output}")
        if payload.get("output_sha256") != sha256(output):
            errors.append(f"sha256 mismatch: {output}")
        sparse = payload.get("sparse")
        if sparse:
            fallback_calls += int(sparse.get("dense_fallback_calls", 0))
            failed_calls += int(sparse.get("failed_calls", 0))
            method = task.get("method")
            if method != "svg2_official_top_p":
                actual = float(sparse["logical_pair_density"])
                target = float(task["density"])
                density_error = abs(actual - target)
                max_density_error = max(max_density_error, density_error)
                if density_error > args.density_tolerance:
                    errors.append(
                        f"density error {density_error:.6g} > {args.density_tolerance}: {stats_path}"
                    )
        completed.append(str(output))
        origin = task.get("result_origin", "stage2_new")
        origins[origin] = origins.get(origin, 0) + 1
    if fallback_calls:
        errors.append(f"dense_fallback_calls={fallback_calls}")
    if failed_calls:
        errors.append(f"failed_calls={failed_calls}")
    if missing and not args.allow_incomplete:
        errors.append(f"missing_tasks={len(missing)}")
    if failed and not args.allow_incomplete:
        errors.append(f"failed_task_files={len(failed)}")
    payload = {
        "suite": str(suite_path),
        "expected_tasks": len(tasks),
        "completed_tasks": len(completed),
        "missing": missing,
        "failed_task_files": failed,
        "errors": errors,
        "fallback_calls": fallback_calls,
        "failed_calls": failed_calls,
        "max_density_error": max_density_error,
        "density_tolerance": args.density_tolerance,
        "result_origins": origins,
        "status": "pass" if not errors else ("partial" if args.allow_incomplete else "fail"),
    }
    output = ROOT / "results" / "manifests" / suite_path.stem.replace(".template", "") / "audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
