#!/usr/bin/env python3
"""Audit the gated 39/120/240 system profile matrix against legacy Final."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def summarize(states_path: Path, expected_path: Path) -> dict:
    states = json.loads(states_path.read_text(encoding="utf-8"))["cases"]
    expected = json.loads(expected_path.read_text(encoding="utf-8"))["cases"]
    expected_by_key = {row["case_key_sha256"]: row for row in expected}
    if len(expected_by_key) != len(expected):
        raise ValueError("expected system profile contains duplicate case keys")
    rows = []
    for state in states:
        key = state.get("case_key_sha256")
        if key not in expected_by_key:
            raise ValueError(f"unexpected system profile case: {key}")
        frozen = expected_by_key[key]
        if state.get("status") not in {"pass", "fail", "negative"}:
            raise ValueError("system profile cases must be terminal")
        service = {
            name: float(state.get(name) or 0.0)
            for name in ("routing_s", "cpu_gather_s", "h2d_s", "attention_s", "rope_s")
        }
        rows.append(
            {
                "case_key_sha256": key,
                "profile_config_id": frozen["profile_config_id"],
                "latent_frames": int(frozen["latent_frames"]),
                "status": state["status"],
                "route_plan_sha256": state.get("route_plan_sha256"),
                "route_plan_sha256s": state.get("route_plan_sha256s", []),
                "end_to_end_s": state.get("end_to_end_s"),
                "service": service,
                "route_gather_h2d_service_s": (
                    service["routing_s"] + service["cpu_gather_s"] + service["h2d_s"]
                ),
                "transferred_bytes": int(state.get("transferred_bytes") or 0),
                "candidate_transfer_bytes": int(state.get("candidate_transfer_bytes") or 0),
                "cache_hit_bytes": int(
                    state.get("stats_summary", {}).get("cache_hit_bytes", 0)
                    if isinstance(state.get("stats_summary"), dict)
                    else 0
                ),
                "cache_miss_bytes": int(
                    state.get("stats_summary", {}).get("cache_miss_bytes", 0)
                    if isinstance(state.get("stats_summary"), dict)
                    else 0
                ),
                "peak_allocated_gb": state.get("peak_allocated_gb"),
            }
        )
    observed_keys = {row["case_key_sha256"] for row in rows}
    missing = sorted(set(expected_by_key) - observed_keys)
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["latent_frames"]][row["profile_config_id"]] = row
    comparisons = []
    route_equivalence = []
    for latent_frames, configs in sorted(grouped.items()):
        legacy = configs.get("legacy")
        if legacy is None or legacy["status"] != "pass":
            continue
        reference_routes = set(legacy["route_plan_sha256s"])
        for config_id, row in sorted(configs.items()):
            if config_id == "legacy" or row["status"] != "pass":
                continue
            service_speedup = (
                legacy["route_gather_h2d_service_s"]
                / row["route_gather_h2d_service_s"]
                if row["route_gather_h2d_service_s"] > 0
                else None
            )
            end_to_end_speedup = (
                float(legacy["end_to_end_s"]) / float(row["end_to_end_s"])
                if row["end_to_end_s"]
                else None
            )
            same_route = set(row["route_plan_sha256s"]) == reference_routes
            route_equivalence.append(same_route)
            comparisons.append(
                {
                    "latent_frames": latent_frames,
                    "profile_config_id": config_id,
                    "same_route_plan_set": same_route,
                    "route_gather_h2d_service_speedup": service_speedup,
                    "end_to_end_speedup": end_to_end_speedup,
                    "transferred_byte_ratio_vs_legacy": (
                        row["transferred_bytes"] / legacy["transferred_bytes"]
                        if legacy["transferred_bytes"]
                        else None
                    ),
                    "peak_allocated_gb": row["peak_allocated_gb"],
                    "cache_hit_bytes": row["cache_hit_bytes"],
                    "cache_miss_bytes": row["cache_miss_bytes"],
                }
            )
    pass_count = sum(row["status"] == "pass" for row in rows)
    fail_count = sum(row["status"] == "fail" for row in rows)
    negative_count = sum(row["status"] == "negative" for row in rows)
    return {
        "status": "pass" if not missing and not fail_count else "incomplete",
        "expected": len(expected),
        "observed": len(rows),
        "pass": pass_count,
        "fail": fail_count,
        "negative": negative_count,
        "missing": len(missing),
        "missing_case_keys": missing,
        "same_route_plan_all_system_comparisons": bool(route_equivalence)
        and all(route_equivalence),
        "comparisons": comparisons,
        "promotion": {
            "pure_system_config": None,
            "status": "pending_nsys_exposed_wait_and_quality_equivalence",
            "service_gate_candidates": [
                row
                for row in comparisons
                if row["same_route_plan_set"]
                and row["route_gather_h2d_service_speedup"] is not None
                and row["route_gather_h2d_service_speedup"] >= 1.10
            ],
        },
        "evidence_boundary": "component service speedup is not measured exposed wait",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = summarize(Path(args.states), Path(args.expected))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "observed": payload["observed"]}, indent=2))


if __name__ == "__main__":
    main()
