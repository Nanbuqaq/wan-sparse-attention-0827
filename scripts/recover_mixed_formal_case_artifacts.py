#!/usr/bin/env python3
"""Recover mixed external/local formal cases into one outer results fact tree."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUCCESS = {"pass", "negative"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_verified(source: Path, destination: Path) -> dict:
    if not source.is_file():
        raise FileNotFoundError(source)
    source_sha = sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256(destination) == source_sha:
        copied = False
    else:
        shutil.copy2(source, destination)
        copied = True
    destination_sha = sha256(destination)
    if destination_sha != source_sha:
        raise RuntimeError(f"recovery SHA mismatch: {destination}")
    return {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "sha256": source_sha,
        "bytes": source.stat().st_size,
        "copied": copied,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    states_path = Path(args.states).resolve()
    expected_path = Path(args.expected).resolve()
    results_root = Path(args.results_root).resolve()
    cases = json.loads(states_path.read_text(encoding="utf-8"))["cases"]
    run_id = args.run_id
    roots = {
        "videos": results_root / "videos" / "recovered" / run_id,
        "latents": results_root / "latents" / "recovered" / run_id,
        "manifests": results_root / "manifests" / run_id,
        "audits": results_root / "audits" / run_id,
        "logs": results_root / "logs" / run_id,
    }
    for path in roots.values():
        path.mkdir(parents=True, exist_ok=True)

    tasks = []
    task_keys = []
    rewritten = []
    for case in cases:
        local = dict(case)
        case_id = case["id"]
        if case.get("status") in SUCCESS:
            video = Path(str(case["video"]))
            latent = Path(str(case.get("latent", video.parent / "latents.pt")))
            stats = Path(str(case["stats"]))
            config = Path(str(case["config"]))
            artifacts = {
                "video": (video, roots["videos"] / case_id / "video.mp4"),
                "latent": (latent, roots["latents"] / case_id / "latents.pt"),
                "stats": (stats, roots["manifests"] / "cases" / case_id / "sparse_history_stats.json"),
                "config": (config, roots["manifests"] / "cases" / case_id / "case_config.json"),
            }
            for key, (source, destination) in artifacts.items():
                tasks.append((source, destination))
                task_keys.append((case_id, key, destination))
            local.update(
                {
                    "video": str(artifacts["video"][1].resolve()),
                    "latent": str(artifacts["latent"][1].resolve()),
                    "stats": str(artifacts["stats"][1].resolve()),
                    "config": str(artifacts["config"][1].resolve()),
                    "recovered_from": str(video.parent.resolve()),
                }
            )
        rewritten.append(local)

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(copy_verified, source, destination) for source, destination in tasks]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda item: item["destination"])

    local_states = {"cases": rewritten}
    local_states_path = roots["manifests"] / "merged_case_states.local.json"
    local_states_path.write_text(
        json.dumps(local_states, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    local_expected = roots["manifests"] / "expected_basic_477.json"
    copy_verified(expected_path, local_expected)
    audit_path = roots["audits"] / "terminal_state_audit.local.json"
    log_path = roots["logs"] / "terminal_state_audit.log"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_case_states.py"),
            "--expected",
            str(local_expected),
            "--states",
            str(local_states_path),
            "--output",
            str(audit_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"recovered case audit failed: {log_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    payload = {
        "status": "pass",
        "run_id": run_id,
        "source_states": str(states_path),
        "source_states_sha256": sha256(states_path),
        "expected": str(expected_path),
        "expected_sha256": sha256(expected_path),
        "cases": len(cases),
        "statuses": dict(Counter(case["status"] for case in cases)),
        "successful_cases": sum(case["status"] in SUCCESS for case in cases),
        "artifact_records": len(records),
        "artifact_bytes": sum(record["bytes"] for record in records),
        "copied_records": sum(record["copied"] for record in records),
        "local_states": str(local_states_path),
        "local_states_sha256": sha256(local_states_path),
        "local_audit": str(audit_path),
        "local_audit_sha256": sha256(audit_path),
        "local_audit_status": audit["status"],
        "records": records,
    }
    recovery_path = roots["audits"] / "recovery_audit.json"
    recovery_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "cases": payload["cases"],
                "statuses": payload["statuses"],
                "artifact_records": payload["artifact_records"],
                "artifact_bytes": payload["artifact_bytes"],
                "recovery_audit": str(recovery_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
