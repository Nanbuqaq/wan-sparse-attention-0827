#!/usr/bin/env python3
"""Select an auditable recovery subset from a frozen expected-case manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", required=True)
    parser.add_argument("--states", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--failure-contains", default="")
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    expected_path = Path(args.expected).resolve()
    states_path = Path(args.states).resolve()
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    states = json.loads(states_path.read_text(encoding="utf-8"))["cases"]
    by_id = {case["id"]: case for case in states}
    if len(by_id) != len(states):
        raise RuntimeError("recovery states contain duplicate case ids")
    selected = []
    for case in expected["cases"]:
        state = by_id.get(case["id"])
        if state is None or state.get("status") != args.status:
            continue
        if args.failure_contains and args.failure_contains not in state.get(
            "failure_reason", ""
        ):
            continue
        if state.get("case_key_sha256") != case.get("case_key_sha256"):
            raise RuntimeError(f"state/expected identity mismatch: {case['id']}")
        selected.append(case)
    if args.expected_count is not None and len(selected) != args.expected_count:
        raise RuntimeError(
            f"selected {len(selected)} recovery cases, expected {args.expected_count}"
        )
    if not selected:
        raise RuntimeError("recovery filter selected no cases")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "commit": expected.get("commit"),
                "cases": selected,
                "recovery_filter": {
                    "states": str(states_path),
                    "status": args.status,
                    "failure_contains": args.failure_contains,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected_cases": len(selected), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
