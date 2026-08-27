#!/usr/bin/env python3
"""Attach matched Dense generation time and end-to-end speedup to case metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from run_matrix import expand_tasks, resolve_common


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--case-metrics", required=True)
    args = parser.parse_args()
    suite = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    suite["common"] = resolve_common(suite["common"])
    dense = {}
    for task in expand_tasks(suite):
        if task["mode"] != "dense":
            continue
        stats_path = (ROOT / task["output"]).with_suffix(".stats.json")
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        dense[(task["prompt_id"], int(task["seed"]))] = float(payload["generation_elapsed_s"])
    path = Path(args.case_metrics)
    if not path.is_absolute():
        path = ROOT / path
    table = pd.read_csv(path)
    table["dense_generation_elapsed_s"] = [dense[(row.prompt_id, int(row.seed))] for row in table.itertuples()]
    table["end_to_end_speedup_vs_dense"] = table["dense_generation_elapsed_s"] / table["generation_elapsed_s"]
    table.to_csv(path, index=False)
    print(json.dumps({"cases": len(table), "dense_references": len(dense), "output": str(path)}, indent=2))


if __name__ == "__main__":
    main()

