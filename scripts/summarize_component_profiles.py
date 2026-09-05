#!/usr/bin/env python3
"""Summarize component service times while keeping critical-path limits explicit."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


COMPONENTS = (
    "routing_s",
    "q_summary_s",
    "d2h_s",
    "cpu_gather_s",
    "h2d_s",
    "rope_s",
    "attention_s",
)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def summarize(paths: list[Path]) -> dict:
    if not paths:
        raise ValueError("component profile summary requires case states")
    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "pass":
            raise ValueError(f"non-pass component profile case: {path}")
        total = float(payload["end_to_end_s"])
        if total <= 0:
            raise ValueError("end_to_end_s must be positive")
        components = {name: float(payload.get(name) or 0.0) for name in COMPONENTS}
        attributed = sum(components.values())
        rows.append(
            {
                "artifact": str(path),
                "prompt_id": payload.get("prompt_id"),
                "seed": payload.get("seed"),
                "latent_frames": payload.get("latent_frames"),
                "end_to_end_s": total,
                "components": components,
                "service_fractions": {
                    name: value / total for name, value in components.items()
                },
                "cpu_route_gather_fraction": (
                    components["routing_s"] + components["cpu_gather_s"]
                )
                / total,
                "unattributed_s": max(0.0, total - attributed),
                "unattributed_fraction": max(0.0, total - attributed) / total,
            }
        )
    names = (*COMPONENTS, "cpu_route_gather", "unattributed")
    aggregate = {}
    for name in names:
        if name == "cpu_route_gather":
            values = [row["cpu_route_gather_fraction"] for row in rows]
        elif name == "unattributed":
            values = [row["unattributed_fraction"] for row in rows]
        else:
            values = [row["service_fractions"][name] for row in rows]
        aggregate[name] = {
            "fraction_median": statistics.median(values),
            "fraction_p95": _percentile(values, 0.95),
            "fraction_min": min(values),
            "fraction_max": max(values),
        }
    attention_below_ten_all = all(
        row["service_fractions"]["attention_s"] < 0.10 for row in rows
    )
    return {
        "status": "pass",
        "cases": len(rows),
        "aggregate": aggregate,
        "preliminary_priority": "cpu_route_gather_then_unattributed_pipeline",
        "kvout_video_gate_currently_justified": not attention_below_ten_all,
        "evidence_boundary": {
            "times_are_component_service_times": True,
            "measured_exposed_wait_available": False,
            "nsys_timeline_available": False,
            "fractions_must_not_be_summed_as_overlapped_critical_path": True,
            "kvout_gate_must_be_rechecked_after_cache_onload_optimization": True,
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = summarize([Path(value) for value in args.input])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "cases": payload["cases"],
                "priority": payload["preliminary_priority"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
