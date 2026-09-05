#!/usr/bin/env python3
"""Aggregate the four-layer early/middle/late capture-system matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(paths: list[Path], *, expected_cases: int = 12) -> dict:
    records = []
    errors = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "pass":
            errors.append(f"non-pass capture analysis: {path}")
            continue
        exact = payload["transfer_layouts"]["exact_compact"]
        row = {
            "artifact": str(path),
            "layer": int(payload["capture_metadata"]["layer"]),
            "current_start": int(payload["capture_metadata"]["current_start"]),
            "history_tokens": int(payload["capture_metadata"]["history_shape"][1]),
            "route_sha": payload["route"]["route_plan_sha256"],
            "history_pair_density": float(payload["route"]["history_pair_density"]),
            "history_transfer_density": float(payload["route"]["history_transfer_density"]),
            "exact_runs": int(exact["source_run_count"]),
            "layouts": {},
        }
        for layout, transfer in payload["transfer_layouts"].items():
            row["layouts"][layout] = {
                "physical_copy_bytes": int(transfer["physical_copy_bytes"]),
                "padding_bytes": int(transfer["padding_bytes"]),
                "source_run_count": int(transfer["source_run_count"]),
                "byte_multiplier_vs_exact": (
                    float(transfer["physical_copy_bytes"])
                    / max(1, int(exact["physical_copy_bytes"]))
                ),
                "run_reduction_vs_exact": (
                    1.0
                    - float(transfer["source_run_count"])
                    / max(1, int(exact["source_run_count"]))
                ),
            }
        records.append(row)
    if len(records) != expected_cases:
        errors.append(f"expected {expected_cases} capture analyses, found {len(records)}")
    aggregate = {}
    for layout in ("block64", "page256", "frame1560"):
        values = [row["layouts"][layout] for row in records]
        if values:
            aggregate[layout] = {
                "byte_multiplier_vs_exact_mean": sum(
                    item["byte_multiplier_vs_exact"] for item in values
                )
                / len(values),
                "run_reduction_vs_exact_mean": sum(
                    item["run_reduction_vs_exact"] for item in values
                )
                / len(values),
                "cases_with_fewer_runs": sum(
                    item["run_reduction_vs_exact"] > 0 for item in values
                ),
            }
    return {
        "status": "pass" if not errors else "incomplete",
        "expected_cases": expected_cases,
        "observed_cases": len(records),
        "errors": errors,
        "aggregate": aggregate,
        "records": sorted(records, key=lambda item: (item["layer"], item["current_start"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--expected-cases", type=int, default=12)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = sorted(Path(args.input_dir).glob("*.json"))
    payload = summarize(paths, expected_cases=args.expected_cases)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "observed": payload["observed_cases"]}, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
