#!/usr/bin/env python3
"""Apply audited prompt/method rules to the formal manual-review table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


SCORE_FIELDS = [
    "subject_identity_1to5",
    "background_consistency_1to5",
    "irreversible_state_reset_count",
    "action_loop_count",
    "action_discontinuity_count",
    "freeze_count",
    "flicker_count",
    "camera_cut_count",
    "late_quarter_quality_1to5",
    "late_quarter_degradation_0to2",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matches(rule: dict, row: dict) -> bool:
    for field in ("prompt_id", "method", "routing_stage"):
        if field in rule and str(rule[field]) != str(row[field]):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    review_path = Path(args.review)
    decisions_path = Path(args.decisions)
    diagnostics_path = Path(args.diagnostics)
    with review_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    diagnostic_cases = {
        case["case_id"]: case
        for case in json.loads(diagnostics_path.read_text(encoding="utf-8"))["cases"]
    }
    defaults = decisions["defaults"]
    rules = decisions.get("rules", [])
    for row in rows:
        decision = dict(defaults)
        notes = []
        if defaults.get("review_notes"):
            notes.append(str(defaults["review_notes"]))
        for rule in rules:
            if matches(rule, row):
                decision.update({field: rule[field] for field in SCORE_FIELDS if field in rule})
                if rule.get("review_notes"):
                    notes.append(str(rule["review_notes"]))
        diagnostic = diagnostic_cases.get(row["case_id"])
        if diagnostic is None:
            raise RuntimeError(f"missing diagnostics: {row['case_id']}")
        decision["freeze_count"] = len(diagnostic["freeze_runs"])
        decision["flicker_count"] = len(diagnostic["flicker_indices"])
        decision["camera_cut_count"] = len(diagnostic["cut_indices"])
        missing = [field for field in SCORE_FIELDS if field not in decision]
        if missing:
            raise RuntimeError(f"incomplete decision {row['case_id']}: {missing}")
        for field in SCORE_FIELDS:
            value = int(decision[field])
            if field in {"subject_identity_1to5", "background_consistency_1to5", "late_quarter_quality_1to5"}:
                if not 1 <= value <= 5:
                    raise ValueError(f"invalid {field}: {value}")
            elif field == "late_quarter_degradation_0to2":
                if not 0 <= value <= 2:
                    raise ValueError(f"invalid {field}: {value}")
            elif value < 0:
                raise ValueError(f"invalid {field}: {value}")
            row[field] = value
        row["reviewer"] = decisions["reviewer"]
        row["review_notes"] = " ".join(notes)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "status": "pass",
        "rows": len(rows),
        "review": str(review_path.resolve()),
        "review_sha256": sha256(review_path),
        "decisions": str(decisions_path.resolve()),
        "decisions_sha256": sha256(decisions_path),
        "diagnostics": str(diagnostics_path.resolve()),
        "diagnostics_sha256": sha256(diagnostics_path),
        "output": str(output.resolve()),
        "output_sha256": sha256(output),
    }
    audit_path = output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
