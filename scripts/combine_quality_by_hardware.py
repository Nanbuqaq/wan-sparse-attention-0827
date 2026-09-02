#!/usr/bin/env python3
"""Select per-case quality metrics from the matching GPU-class reference run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--source", action="append", required=True, help="GPU_CLASS=SUMMARY_JSON")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    hardware_path = Path(args.hardware).resolve()
    with hardware_path.open(newline="", encoding="utf-8") as handle:
        hardware = {row["case_id"]: row for row in csv.DictReader(handle)}
    metrics_by_gpu: dict[str, dict[str, dict]] = {}
    sources = []
    for value in args.source:
        gpu_class, separator, path_value = value.partition("=")
        if not separator or not gpu_class or not path_value:
            raise ValueError(f"invalid source: {value}")
        path = Path(path_value).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        target = metrics_by_gpu.setdefault(gpu_class, {})
        for case_id, metrics in payload.get("candidates", {}).items():
            if case_id in target and target[case_id] != metrics:
                raise RuntimeError(f"conflicting metrics for {gpu_class}: {case_id}")
            target[case_id] = metrics
        sources.append({"gpu_class": gpu_class, "path": str(path), "sha256": sha256(path)})
    candidates = {}
    selection = {}
    missing = []
    for case_id, row in hardware.items():
        if row["status"] not in {"pass", "negative"}:
            continue
        gpu_class = row["gpu_class"]
        metrics = metrics_by_gpu.get(gpu_class, {}).get(case_id)
        if metrics is None:
            missing.append({"case_id": case_id, "gpu_class": gpu_class})
            continue
        candidates[case_id] = metrics
        selection[case_id] = gpu_class
    if missing:
        raise RuntimeError(f"missing same-hardware quality metrics: {missing[:8]}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "pass",
        "hardware_provenance": {
            "path": str(hardware_path),
            "sha256": sha256(hardware_path),
        },
        "sources": sources,
        "selection": selection,
        "candidates": candidates,
        "case_count": len(candidates),
        "rule": "each candidate uses the Dense quality run matching its own gpu_class",
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "cases": len(candidates), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
