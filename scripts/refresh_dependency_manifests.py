#!/usr/bin/env python3
"""Recompute task-scoped dependency manifests after audit-schema-only changes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.dependencies import build_execution_dependency_manifest, generation_fingerprint, task_fingerprint
from run_matrix import expand_tasks, resolve_common


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    args = parser.parse_args()
    suite_path = Path(args.suite).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite["common"] = resolve_common(suite["common"])
    from diffusers import WanPipeline
    from diffusers.schedulers import UniPCMultistepScheduler

    updated = 0
    for task in expand_tasks(suite):
        stats_path = (ROOT / task["output"]).with_suffix(".stats.json")
        if not stats_path.is_file():
            continue
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed" or task.get("result_origin") == "stage1_reused":
            continue
        payload["task_fingerprint"] = task_fingerprint(task, suite["common"])
        payload["generation_fingerprint"] = generation_fingerprint(task, suite["common"])
        payload["execution_dependency_manifest"] = build_execution_dependency_manifest(
            task,
            suite["common"],
            pipeline_class=WanPipeline,
            scheduler_class=UniPCMultistepScheduler,
        )
        stats_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        updated += 1
    print(json.dumps({"suite": str(suite_path), "updated": updated}, indent=2))


if __name__ == "__main__":
    main()

