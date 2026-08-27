#!/usr/bin/env python3
"""Ensure every expected GPU case reaches pass/fail/negative with valid evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TERMINAL = {"pass", "fail", "negative"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", required=True)
    parser.add_argument("--states", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))["cases"]
    states = json.loads(Path(args.states).read_text(encoding="utf-8"))["cases"]
    by_id = {case["id"]: case for case in states}
    errors = []
    records = []
    for case in expected:
        state = by_id.get(case["id"])
        if state is None:
            errors.append(f"missing terminal state: {case['id']}")
            continue
        status = state.get("status")
        if status not in TERMINAL:
            errors.append(f"non-terminal state {status!r}: {case['id']}")
        if status in {"pass", "negative"}:
            required = ["backend", "stats", "config"]
            if state.get("routing_stage") != "N/A":
                required.append("route_plan_sha256")
            for key in required:
                if not state.get(key):
                    errors.append(f"successful case missing {key}: {case['id']}")
            if state.get("failed_calls", 0) or state.get("fallback_calls", 0) or state.get("nan_calls", 0):
                errors.append(f"successful case has failed/fallback/NaN calls: {case['id']}")
            if state.get("backend") in {"unknown", "fallback", "dense_fallback"}:
                errors.append(f"successful case did not hit a verified backend: {case['id']}")
            stats_path = Path(str(state.get("stats", "")))
            if stats_path.is_file():
                stats = json.loads(stats_path.read_text(encoding="utf-8"))
                for key in ("history_density", "global_executed_density"):
                    value = stats.get(key)
                    if value is None or not 0.0 <= float(value) <= 1.0:
                        errors.append(f"invalid {key} denominator/value: {case['id']}")
                transfer_density = stats.get("history_transfer_density")
                if transfer_density is not None and not 0.0 <= float(transfer_density) <= 1.0:
                    errors.append(f"invalid history_transfer_density: {case['id']}")
                if stats.get("failed_calls", 0) or stats.get("dense_fallback_calls", 0):
                    errors.append(f"stats contain failed/fallback calls: {case['id']}")
        if status == "fail" and not state.get("failure_reason"):
            errors.append(f"failed case missing reason: {case['id']}")
        records.append(state)
    payload = {
        "status": "pass" if not errors else "fail",
        "expected_cases": len(expected),
        "terminal_cases": len(records),
        "errors": errors,
        "cases": records,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
