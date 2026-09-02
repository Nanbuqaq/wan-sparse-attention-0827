#!/usr/bin/env python3
"""Attach auditable GPU-class provenance to mixed formal case states."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True)
    parser.add_argument("--mapping", action="append", required=True, help="PATH_PREFIX=GPU_CLASS")
    parser.add_argument(
        "--failure-mapping",
        action="append",
        default=[],
        help="FAILURE_SUBSTRING=GPU_CLASS for failures without artifact paths",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    mappings = []
    for value in args.mapping:
        prefix, separator, gpu_class = value.partition("=")
        if not separator or not prefix or not gpu_class:
            raise ValueError(f"invalid mapping: {value}")
        mappings.append((str(Path(prefix).resolve()), gpu_class))
    failure_mappings = []
    for value in args.failure_mapping:
        substring, separator, gpu_class = value.partition("=")
        if not separator or not substring or not gpu_class:
            raise ValueError(f"invalid failure mapping: {value}")
        failure_mappings.append((substring, gpu_class))
    states_path = Path(args.states).resolve()
    cases = json.loads(states_path.read_text(encoding="utf-8"))["cases"]
    rows = []
    for case in cases:
        evidence_values = [
            str(case.get("video") or ""),
            str(case.get("config") or ""),
            str(case.get("stats") or ""),
        ]
        matches = [
            (prefix, gpu_class, value)
            for prefix, gpu_class in mappings
            for value in evidence_values
            if value and str(Path(value).resolve()).startswith(prefix)
        ]
        classes = {gpu_class for _, gpu_class, _ in matches}
        failure_reason = str(case.get("failure_reason") or "")
        classes.update(
            gpu_class
            for substring, gpu_class in failure_mappings
            if substring in failure_reason
        )
        if len(classes) != 1:
            raise RuntimeError(
                f"hardware mapping is not unique for {case['id']}: {sorted(classes)}"
            )
        gpu_class = next(iter(classes))
        evidence = next(
            (value for _, candidate, value in matches if candidate == gpu_class),
            f"failure_reason:{failure_reason[:160]}",
        )
        rows.append(
            {
                "case_id": case["id"],
                "case_key_sha256": case["case_key_sha256"],
                "method": case["method"],
                "prompt_id": case["prompt_id"],
                "status": case["status"],
                "gpu_class": gpu_class,
                "hardware_evidence_path": evidence,
                "speed_group": f"{gpu_class}|{case.get('runtime')}|{case.get('backend')}",
            }
        )
    rows.sort(key=lambda row: row["case_id"])
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "status": "pass",
        "states": str(states_path),
        "states_sha256": sha256(states_path),
        "cases": len(rows),
        "gpu_counts": dict(Counter(row["gpu_class"] for row in rows)),
        "status_by_gpu": {
            gpu: dict(Counter(row["status"] for row in rows if row["gpu_class"] == gpu))
            for gpu in sorted({row["gpu_class"] for row in rows})
        },
        "csv": str(csv_path.resolve()),
        "csv_sha256": sha256(csv_path),
        "mappings": [{"prefix": prefix, "gpu_class": gpu} for prefix, gpu in mappings],
        "failure_mappings": [
            {"substring": substring, "gpu_class": gpu}
            for substring, gpu in failure_mappings
        ],
    }
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
