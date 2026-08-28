#!/usr/bin/env python3
"""Audit terminal early/middle/late selected-method backend replay states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SNAPSHOTS = ("early", "middle", "late")
TERMINAL = {"pass", "negative", "fail"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--states", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    methods = json.loads(Path(args.selection).read_text(encoding="utf-8"))[
        "selected_methods"
    ]
    expected = {f"{method}__{snapshot}" for method in methods for snapshot in SNAPSHOTS}
    merged = {}
    errors = []
    for value in args.states:
        path = Path(value)
        if not path.is_file():
            continue
        for record in json.loads(path.read_text(encoding="utf-8"))["cases"]:
            case_id = record["id"]
            if case_id in merged:
                errors.append(f"duplicate route benchmark state: {case_id}")
            merged[case_id] = record
    for case_id in sorted(expected - set(merged)):
        method, snapshot = case_id.rsplit("__", 1)
        merged[case_id] = {
            "id": case_id,
            "method": method,
            "snapshot": snapshot,
            "status": "fail",
            "failure_reason": "route benchmark runner emitted no terminal state",
        }
    for case_id in sorted(set(merged) - expected):
        errors.append(f"unexpected route benchmark state: {case_id}")
    for case_id in sorted(expected):
        record = merged[case_id]
        if record.get("status") not in TERMINAL:
            errors.append(f"non-terminal route benchmark state: {case_id}")
            continue
        if record.get("status") == "fail":
            if not record.get("failure_reason") and record.get("subprocess_status") == 0:
                errors.append(f"failed route benchmark lacks evidence: {case_id}")
            continue
        if not record.get("route_plan_sha256"):
            errors.append(f"route benchmark missing route SHA: {case_id}")
        benchmark_path = Path(str(record.get("benchmark", "")))
        if not benchmark_path.is_file():
            errors.append(f"route benchmark missing JSON artifact: {case_id}")
            continue
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        backends = benchmark.get("backends", {})
        if set(backends) != {"grouped_fa2", "fixed64_rect", "varlen_triton"}:
            errors.append(f"route benchmark backend set incomplete: {case_id}")
            continue
        route_shas = {item.get("route_plan_sha256") for item in backends.values()}
        if route_shas != {record.get("route_plan_sha256")}:
            errors.append(f"route benchmark did not reuse one route SHA: {case_id}")
        for backend, item in backends.items():
            if item.get("warmup") != 5 or item.get("iterations") != 20:
                errors.append(f"route benchmark is not warm 5+20: {case_id}/{backend}")
            for field in (
                "cold_wall_ms",
                "cold_backend_ms",
                "wall_ms_median",
                "backend_ms_median",
            ):
                if item.get(field) is None:
                    errors.append(f"route benchmark missing {field}: {case_id}/{backend}")
    records = [merged[case_id] for case_id in sorted(expected)]
    payload = {
        "status": "pass" if not errors else "fail",
        "expected_cases": len(expected),
        "terminal_cases": len(records),
        "pass_cases": sum(item.get("status") == "pass" for item in records),
        "negative_cases": sum(item.get("status") == "negative" for item in records),
        "fail_cases": sum(item.get("status") == "fail" for item in records),
        "errors": errors,
        "cases": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
