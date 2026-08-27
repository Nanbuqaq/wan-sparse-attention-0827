#!/usr/bin/env python3
"""Verify cross-backend comparisons reuse identical sparse graphs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_matrix import expand_tasks, resolve_common


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    args = parser.parse_args()
    suite_path = Path(args.suite).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite["common"] = resolve_common(suite["common"])
    records = {}
    for task in expand_tasks(suite):
        if task["matrix_id"] != "kernel_cross_backend_d250":
            continue
        stats_path = (ROOT / task["output"]).with_suffix(".stats.json")
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        key = (task["prompt_id"], int(task["seed"]), task.get("graph_kind"), task.get("route_family"))
        records.setdefault(key, {})[task["backend"]] = (payload.get("sparse") or {}).get("route_graph_hashes", {})
    rows = []
    for key, backends in sorted(records.items()):
        hashes = list(backends.values())
        equal = len(hashes) == 2 and hashes[0] == hashes[1] and bool(hashes[0])
        rows.append(
            {
                "prompt_id": key[0],
                "seed": key[1],
                "graph_kind": key[2],
                "route_family": key[3],
                "backends": backends,
                "identical": equal,
            }
        )
    payload = {
        "schema_version": 2,
        "rows": rows,
        "status": (
            "pass"
            if rows and all(row["identical"] for row in rows)
            else "dynamic_graph_divergence_detected"
        ),
        "eligible_for_pure_kernel_ranking": bool(rows) and all(row["identical"] for row in rows),
        "note": "Independent end-to-end backend runs may diverge after layer 0 because backend numerical differences change later routing inputs. Use the captured same-RoutePlan replay benchmark for pure kernel ranking.",
    }
    output = ROOT / "results/manifests/formal_stage2_v2/route_graph_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "comparisons": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
