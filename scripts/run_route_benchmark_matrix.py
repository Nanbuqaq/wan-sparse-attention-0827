#!/usr/bin/env python3
"""Run a disjoint shard of selected-method early/middle/late route replays."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = {
    "early": "layer00_start00028080.pt",
    "middle": "layer00_start00093600.pt",
    "late": "layer00_start00177840.pt",
}


def classify(payload: dict) -> tuple[str, list[dict]]:
    backends = payload.get("backends", {})
    if set(backends) != {"grouped_fa2", "fixed64_rect", "varlen_triton"}:
        return "fail", []
    if any(not record.get("same_route_plan") for record in backends.values()):
        return "fail", []
    if backends["grouped_fa2"].get("status") != "pass":
        return "fail", []
    negatives = []
    for backend in ("fixed64_rect", "varlen_triton"):
        if backends[backend].get("status") != "pass":
            negatives.append(
                {
                    "backend": backend,
                    "reason": "frozen different-kernel numerical threshold exceeded",
                    "error_vs_grouped": backends[backend].get("error_vs_grouped"),
                }
            )
    return ("negative" if negatives else "pass"), negatives


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--method-params", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cache-root", default="/tmp/wan_longlive_pareto_route_bench")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")

    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    methods = list(selection.get("selected_methods", []))
    if not methods:
        raise ValueError("empty Pareto selection")
    capture_dir = Path(args.capture_dir)
    captures = {name: capture_dir / filename for name, filename in SNAPSHOTS.items()}
    missing = [str(path) for path in captures.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing route snapshots: {missing}")
    all_tasks = [(method, snapshot) for method in methods for snapshot in SNAPSHOTS]
    tasks = all_tasks[args.shard_index :: args.shard_count]
    if not tasks:
        raise ValueError("empty route benchmark shard")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for method, snapshot in tasks:
        task_id = f"{method}__{snapshot}"
        task_root = output_root / task_id
        cache_root = Path(args.cache_root) / task_id
        (cache_root / "triton").mkdir(parents=True, exist_ok=True)
        (cache_root / "torchinductor").mkdir(parents=True, exist_ok=True)
        task_root.mkdir(parents=True, exist_ok=True)
        output = task_root / "benchmark.json"
        environment = os.environ.copy()
        environment["TRITON_CACHE_DIR"] = str(cache_root / "triton")
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "torchinductor")
        command = [
            sys.executable,
            str(ROOT / "scripts/benchmark_route_backends.py"),
            "--capture",
            str(captures[snapshot]),
            "--output",
            str(output),
            "--method",
            method,
            "--density",
            "0.25",
            "--warmup",
            "5",
            "--iterations",
            "20",
            "--method-params-file",
            args.method_params,
        ]
        with (task_root / "benchmark.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                check=False,
            )
        if output.is_file():
            payload = json.loads(output.read_text(encoding="utf-8"))
            status, negatives = classify(payload)
            records.append(
                {
                    "id": task_id,
                    "method": method,
                    "snapshot": snapshot,
                    "capture_start": payload.get("capture_metadata", {}).get(
                        "current_start"
                    ),
                    "status": status,
                    "subprocess_status": completed.returncode,
                    "route_plan_sha256": payload.get("route", {}).get(
                        "route_plan_sha256"
                    ),
                    "kernel_negatives": negatives,
                    "benchmark": str(output),
                }
            )
        else:
            records.append(
                {
                    "id": task_id,
                    "method": method,
                    "snapshot": snapshot,
                    "status": "fail",
                    "subprocess_status": completed.returncode,
                    "failure_reason": "benchmark subprocess emitted no JSON",
                }
            )
    payload = {
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "cases": records,
    }
    (output_root / f"shard_{args.shard_index}_states.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"tasks": len(records), "statuses": [item["status"] for item in records]}, indent=2))
    if any(item["status"] == "fail" for item in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
