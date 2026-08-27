#!/usr/bin/env python3
"""Reuse any completed task whose scoped execution dependencies and config match."""

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
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["common"] = resolve_common(payload["common"])
    return payload


def reuse_key(task: dict, common: dict) -> str:
    ignored = {"id", "base_method_id", "matrix_id", "output", "result_origin"}
    return canonical_json(
        {
            "task": {key: value for key, value in task.items() if key not in ignored},
            "generation": generation_fingerprint(task, common),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-suite", required=True)
    parser.add_argument("--target-suite", required=True)
    parser.add_argument("--skip-method", action="append", default=[])
    args = parser.parse_args()
    source_path = Path(args.source_suite).resolve()
    target_path = Path(args.target_suite).resolve()
    source = load_suite(source_path)
    target = load_suite(target_path)
    from diffusers import WanPipeline
    from diffusers.schedulers import UniPCMultistepScheduler

    available = {}
    for task in expand_tasks(source):
        video = ROOT / task["output"]
        stats = video.with_suffix(".stats.json")
        if not video.is_file() or not stats.is_file():
            continue
        payload = json.loads(stats.read_text(encoding="utf-8"))
        if payload.get("status") == "completed":
            available[reuse_key(task, source["common"])] = (video, stats, payload)
    imported = []
    for task in expand_tasks(target):
        if task.get("method") in set(args.skip_method):
            continue
        key = reuse_key(task, target["common"])
        if key not in available:
            continue
        source_video, _, old = available[key]
        current_manifest = build_execution_dependency_manifest(
            task,
            target["common"],
            pipeline_class=WanPipeline,
            scheduler_class=UniPCMultistepScheduler,
        )
        old_manifest = old["execution_dependency_manifest"]
        for field in ("generation", "runtime", "dependencies"):
            if old_manifest.get(field) != current_manifest.get(field):
                raise RuntimeError(f"dependency mismatch in {field}: {source_video}")
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
            "result_origin": "stage2_reused",
            "reuse_source": str(source_video),
        }
        destination.with_suffix(".stats.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        imported.append({"source": str(source_video), "destination": str(destination)})
    print(json.dumps({"imported": len(imported), "items": imported}, indent=2))


if __name__ == "__main__":
    main()
