#!/usr/bin/env python3
"""Summarize H2D/layout replay while preserving source-preparation limits."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def summarize(paths: list[Path], *, cost_model_path: Path | None = None) -> dict:
    if not paths:
        raise ValueError("transfer replay summary requires benchmark artifacts")
    rows = []
    hardware = set()
    source_preparation = []
    fastest = Counter()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "pass":
            raise ValueError(f"non-pass transfer replay: {path}")
        identity = (payload.get("gpu"), tuple(payload.get("compute_capability", ())))
        hardware.add(identity)
        source_preparation.append(float(payload.get("source_preparation_s", 0.0)))
        results = payload["results"]
        winner = min(
            results,
            key=lambda key: float(results[key]["gather_plus_h2d_s_median"]),
        )
        fastest[winner] += 1
        for case_id, record in results.items():
            rows.append(
                {
                    "artifact": str(path),
                    "capture": payload["capture"],
                    "case_id": case_id,
                    "layout": record["layout"],
                    "transfer_mode": record["transfer_mode"],
                    "staging_mode": record.get("staging_mode"),
                    "gather_plus_h2d_s_median": float(
                        record["gather_plus_h2d_s_median"]
                    ),
                    "cpu_gather_s_median": float(record["cpu_gather_s_median"]),
                    "h2d_s_median": float(record["h2d_s_median"]),
                    "transferred_bytes": int(record["transferred_bytes"]),
                    "payload_bytes": int(record["payload_bytes"]),
                    "source_run_count": int(record["source_run_count"]),
                    "h2d_copy_count": int(record["h2d_copy_count"]),
                }
            )
    if len(hardware) != 1:
        raise ValueError("one transfer summary cannot mix hardware identities")
    by_case = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    aggregate = {}
    for case_id, selected in sorted(by_case.items()):
        aggregate[case_id] = {
            "captures": len(selected),
            "median_gather_plus_h2d_s": statistics.median(
                row["gather_plus_h2d_s_median"] for row in selected
            ),
            "median_cpu_gather_s": statistics.median(
                row["cpu_gather_s_median"] for row in selected
            ),
            "median_h2d_s": statistics.median(
                row["h2d_s_median"] for row in selected
            ),
            "mean_byte_multiplier_vs_payload": sum(
                row["transferred_bytes"] / max(1, row["payload_bytes"])
                for row in selected
            )
            / len(selected),
            "fastest_capture_count": fastest[case_id],
        }
    cost_model = None
    if cost_model_path is not None:
        cost_model = json.loads(cost_model_path.read_text(encoding="utf-8"))
    gpu, capability = next(iter(hardware))
    return {
        "status": "pass",
        "gpu": gpu,
        "compute_capability": list(capability),
        "captures": len(paths),
        "cases_per_capture": len(by_case),
        "median_one_time_candidate_pin_s": statistics.median(source_preparation),
        "fastest_case_counts": dict(sorted(fastest.items())),
        "aggregate": aggregate,
        "cost_model_gate": (
            {
                "status": cost_model["status"],
                "heldout_mape": cost_model["heldout_mape"],
                "mape_gate": cost_model["mape_gate"],
                "cost_aware_admission_allowed": cost_model[
                    "cost_aware_admission_allowed"
                ],
            }
            if cost_model is not None
            else None
        ),
        "promotion": {
            "pure_system_layout_promoted": False,
            "reason": (
                "direct multi-run excludes one-time candidate pinning and packed modes "
                "include CPU gather; end-to-end exposed wait is not yet measured"
            ),
            "cost_aware_admission_promoted": bool(
                cost_model and cost_model["cost_aware_admission_allowed"]
            ),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--cost-model")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = summarize(
        [Path(value) for value in args.input],
        cost_model_path=Path(args.cost_model) if args.cost_model else None,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "gpu": payload["gpu"],
                "captures": payload["captures"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
