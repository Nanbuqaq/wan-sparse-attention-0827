#!/usr/bin/env python3
"""Enforce the no-training-before-complete-training-free-evidence gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-audit", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--rule", default="configs/training_gate.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = json.loads(Path(args.base_audit).read_text(encoding="utf-8"))
    diagnostics = json.loads(Path(args.diagnostics).read_text(encoding="utf-8"))
    rule = json.loads(Path(args.rule).read_text(encoding="utf-8"))
    base_complete = (
        base.get("status") == "pass"
        and int(base.get("expected_cases", -1)) == rule["required_base_cases"]
        and int(base.get("terminal_cases", -1)) == rule["required_base_cases"]
        and not base.get("errors")
    )
    conditions = {
        name: bool(diagnostics.get(name)) for name in rule["required_conditions"]
    }
    unmet = [name for name, passed in conditions.items() if not passed]
    if not base_complete:
        unmet.insert(0, "all_training_free_base_cases_terminal_and_audited")
    triggered = base_complete and not unmet
    payload = {
        "status": "pass",
        "decision": "train_20_step_output_mse" if triggered else "do_not_train",
        "training_triggered": triggered,
        "base_complete": base_complete,
        "conditions": conditions,
        "unmet_conditions": unmet,
        "protocol": rule["triggered_protocol"] if triggered else None,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
