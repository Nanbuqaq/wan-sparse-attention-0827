#!/usr/bin/env python3
"""Reuse Dense artifacts across suites when task-scoped execution dependencies match."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.dependencies import build_execution_dependency_manifest, canonical_json, generation_fingerprint, sha256_file, task_fingerprint
from run_matrix import expand_tasks, resolve_common


def load_suite(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    value["common"] = resolve_common(value["common"])
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-suite", required=True)
    parser.add_argument("--target-suite", required=True)
    args = parser.parse_args()
    source_path = Path(args.source_suite).resolve()
    target_path = Path(args.target_suite).resolve()
    source = load_suite(source_path)
    target = load_suite(target_path)
    from diffusers import WanPipeline
    from diffusers.schedulers import UniPCMultistepScheduler

    source_dense = {}
    for task in expand_tasks(source):
        if task["mode"] != "dense":
            continue
        video = ROOT / task["output"]
        stats = video.with_suffix(".stats.json")
        if video.is_file() and stats.is_file():
            source_dense[canonical_json(generation_fingerprint(task, source["common"]))] = (video, stats)
    imported = []
    for task in expand_tasks(target):
        if task["mode"] != "dense":
            continue
        key = canonical_json(generation_fingerprint(task, target["common"]))
        if key not in source_dense:
            continue
        source_video, source_stats = source_dense[key]
        old = json.loads(source_stats.read_text(encoding="utf-8"))
        current_manifest = build_execution_dependency_manifest(
            task,
            target["common"],
            pipeline_class=WanPipeline,
            scheduler_class=UniPCMultistepScheduler,
        )
        old_manifest = old.get("execution_dependency_manifest") or {}
        for field in ("generation", "runtime", "dependencies"):
            if old_manifest.get(field) != current_manifest.get(field):
                raise RuntimeError(f"Dense dependency mismatch in {field}: {source_video}")
        destination = ROOT / task["output"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            os.link(source_video, destination)
        elif sha256_file(destination) != sha256_file(source_video):
            raise RuntimeError(f"destination differs: {destination}")
        payload = {
            **old,
            "task": task,
            "common": target["common"],
            "suite": str(target_path),
            "suite_sha256": sha256_file(target_path),
            "task_fingerprint": task_fingerprint(task, target["common"]),
            "generation_fingerprint": generation_fingerprint(task, target["common"]),
            "execution_dependency_manifest": current_manifest,
            "output": str(destination.resolve()),
            "output_sha256": sha256_file(destination),
            "result_origin": "stage1_reused" if task.get("result_origin") == "stage1_reused" else "stage2_reused",
            "reuse_source": str(source_video),
        }
        destination.with_suffix(".stats.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        imported.append({"source": str(source_video), "destination": str(destination)})
    print(json.dumps({"imported": len(imported), "items": imported}, indent=2))


if __name__ == "__main__":
    main()

