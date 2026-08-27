#!/usr/bin/env python3
"""Restartable local Wan sparse-attention video matrix runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from adapters import MethodConfig, SparseRunStats, install_sparse_processors
from adapters.dependencies import (
    build_execution_dependency_manifest,
    generation_fingerprint,
    task_fingerprint,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def density_tag(value: float) -> str:
    return f"d{int(round(value * 1000)):03d}"


def resolve_common(common: dict) -> dict:
    output = dict(common)
    for key, value in list(output.items()):
        if isinstance(value, str):
            expanded = os.path.expandvars(value)
            if "${" in expanded:
                raise RuntimeError(f"unresolved environment variable in common.{key}: {value}")
            output[key] = expanded
    return output


def expand_tasks(suite: dict) -> list[dict]:
    prompts = {item["id"]: item for item in suite["prompts"]}
    methods = {item["id"]: item for item in suite["methods"]}
    output_root = suite.get("output_root", "results/videos")
    tasks: list[dict] = []
    seen: set[tuple] = set()
    for matrix in suite["matrices"]:
        densities = matrix.get("densities", [None])
        for prompt_id in matrix["prompt_ids"]:
            for seed in matrix["seeds"]:
                for method_id in matrix["method_ids"]:
                    method = dict(methods[method_id])
                    method_densities = [None] if method["mode"] == "dense" else densities
                    for density in method_densities:
                        identity = (matrix["id"], prompt_id, int(seed), method_id, density)
                        if identity in seen:
                            continue
                        seen.add(identity)
                        task_id = method_id if density is None else f"{method_id}_{density_tag(float(density))}"
                        task = {
                            **method,
                            "id": task_id,
                            "base_method_id": method_id,
                            "matrix_id": matrix["id"],
                            "prompt_id": prompt_id,
                            "prompt": prompts[prompt_id]["prompt"],
                            "seed": int(seed),
                        }
                        if "result_origin" in matrix:
                            task["result_origin"] = matrix["result_origin"]
                        if density is not None:
                            task["density"] = float(density)
                        task["output"] = (
                            f"{output_root}/{matrix['id']}/{prompt_id}/"
                            f"seed_{int(seed):06d}/{task_id}.mp4"
                        )
                        tasks.append(task)
    return tasks


def method_config(task: dict, common: dict) -> MethodConfig:
    return MethodConfig(
        method=task["method"],
        backend=task["backend"],
        density=float(task.get("density", 0.25)),
        parameter_origin=task.get("parameter_origin", "exact_budget_1p3b_480p"),
        q_clusters=int(task.get("q_clusters", 300)),
        k_clusters=int(task.get("k_clusters", 1000)),
        kmeans_init_iterations=int(task.get("kmeans_init_iterations", 50)),
        kmeans_step_iterations=int(task.get("kmeans_step_iterations", 2)),
        cluster_seed=int(task.get("cluster_seed", 42)),
        block_size=int(task.get("block_size", 64)),
        top_p=float(task.get("top_p", 0.9)),
        min_k_ratio=float(task.get("min_k_ratio", 0.10)),
        official_first_timestep_fraction=float(
            task.get("official_first_timestep_fraction", 0.20)
        ),
        official_first_layer_fraction=float(task.get("official_first_layer_fraction", 0.03)),
        inference_steps=int(task.get("steps", common["steps"])),
        calls_per_step=int(task.get("calls_per_step", 2)),
        measure_timing=bool(task.get("measure_timing", True)),
        route_params=dict(task.get("route_params", {})),
        backend_params=dict(task.get("backend_params", {})),
    )


def run_generation_task(pipe, task: dict, common: dict, generator: torch.Generator):
    """The generation call hashed into each task dependency manifest."""
    return pipe(
        prompt=task["prompt"],
        height=int(task.get("height", common["height"])),
        width=int(task.get("width", common["width"])),
        num_frames=int(task.get("frames", common["frames"])),
        num_inference_steps=int(task.get("steps", common["steps"])),
        guidance_scale=float(task.get("guidance", common["guidance"])),
        generator=generator,
        output_type="np",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--matrix", action="append", default=[])
    parser.add_argument("--output-root")
    parser.add_argument("--manifest-root")
    args = parser.parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard arguments")

    suite_path = Path(args.suite).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite["common"] = resolve_common(suite["common"])
    if args.output_root:
        suite["output_root"] = str(Path(args.output_root).resolve())
    if args.manifest_root:
        suite["manifest_root"] = str(Path(args.manifest_root).resolve())
    all_tasks = expand_tasks(suite)
    if args.include:
        all_tasks = [
            task for task in all_tasks if any(value in task["id"] for value in args.include)
        ]
    if args.prompt:
        all_tasks = [task for task in all_tasks if task["prompt_id"] in set(args.prompt)]
    if args.matrix:
        all_tasks = [task for task in all_tasks if task["matrix_id"] in set(args.matrix)]
    tasks = [
        task
        for index, task in enumerate(all_tasks)
        if index % args.num_shards == args.shard_index
    ]
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    from diffusers import WanPipeline
    from diffusers.utils import export_to_video

    common = suite["common"]
    pipe = WanPipeline.from_pretrained(common["model"], torch_dtype=torch.bfloat16)
    if hasattr(pipe.scheduler, "shift"):
        pipe.scheduler.shift = common["shift"]
    native_processors = dict(pipe.transformer.attn_processors)
    pipe.enable_model_cpu_offload()

    rows = []
    wall_start = time.time()
    for task_index, task in enumerate(tasks):
        output = ROOT / task["output"]
        stats_path = output.with_suffix(".stats.json")
        error_path = output.with_suffix(".error.json")
        dependency_manifest = build_execution_dependency_manifest(
            task,
            common,
            pipeline_class=type(pipe),
            scheduler_class=type(pipe.scheduler),
        )
        if output.is_file() and stats_path.is_file():
            existing = json.loads(stats_path.read_text(encoding="utf-8"))
            existing_hash = (existing.get("execution_dependency_manifest") or {}).get(
                "task_execution_hash"
            )
            if (
                existing.get("status") == "completed"
                and existing_hash == dependency_manifest["task_execution_hash"]
            ):
                print(f"[skip] {task_index + 1}/{len(tasks)} {output}", flush=True)
                rows.append({"output": str(output), "status": "skipped"})
                continue
            raise RuntimeError(
                "refusing to overwrite an existing artifact with a different or legacy "
                f"execution dependency hash: {output}; use a new suite-v2 output_root"
            )

        pipe.transformer.set_attn_processor(dict(native_processors))
        sparse_stats = None
        try:
            if task["mode"] == "sparse":
                sparse_stats = SparseRunStats()
                install_sparse_processors(
                    pipe.transformer,
                    config=method_config(task, common),
                    stats=sparse_stats,
                )
            elif task["mode"] != "dense":
                raise ValueError(f"unknown task mode: {task['mode']}")

            generator = torch.Generator(device="cpu").manual_seed(task["seed"])
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = run_generation_task(pipe, task, common, generator)
            torch.cuda.synchronize()
            generation_s = time.perf_counter() - start
            output.parent.mkdir(parents=True, exist_ok=True)
            export_start = time.perf_counter()
            export_to_video(result.frames[0], str(output), fps=int(task.get("fps", common["fps"])))
            export_s = time.perf_counter() - export_start
            payload = {
                "status": "completed",
                "task": task,
                "task_fingerprint": task_fingerprint(task, common),
                "generation_fingerprint": generation_fingerprint(task, common),
                "execution_dependency_manifest": dependency_manifest,
                "common": common,
                "suite": str(suite_path),
                "suite_sha256": sha256(suite_path),
                "output": str(output.resolve()),
                "output_sha256": sha256(output),
                "generation_elapsed_s": generation_s,
                "export_elapsed_s": export_s,
                "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
                "sparse": sparse_stats.as_dict() if sparse_stats else None,
                "runtime": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
                    "visible_device_count": torch.cuda.device_count(),
                },
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            }
            stats_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            error_path.unlink(missing_ok=True)
            rows.append({"output": str(output), "status": "completed", "elapsed_s": generation_s})
            print(
                f"[done] {task_index + 1}/{len(tasks)} {task['prompt_id']} {task['id']} "
                f"generation={generation_s:.2f}s",
                flush=True,
            )
        except Exception as error:
            output.parent.mkdir(parents=True, exist_ok=True)
            error_payload = {
                "status": "failed",
                "task": task,
                "task_fingerprint": task_fingerprint(task, common),
                "generation_fingerprint": generation_fingerprint(task, common),
                "execution_dependency_manifest": dependency_manifest,
                "suite": str(suite_path),
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "sparse": sparse_stats.as_dict() if sparse_stats else None,
            }
            error_path.write_text(
                json.dumps(error_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            rows.append({"output": str(output), "status": "failed", "error": repr(error)})
            print(f"[failed] {task_index + 1}/{len(tasks)} {task['id']}: {error!r}", flush=True)
            if args.fail_fast:
                raise
        finally:
            torch.cuda.empty_cache()

    manifest_root = ROOT / suite.get("manifest_root", "results/manifests/run")
    manifest_root.mkdir(parents=True, exist_ok=True)
    summary = manifest_root / f"shard_{args.shard_index:02d}_run_summary.json"
    summary.write_text(
        json.dumps(
            {
                "suite": str(suite_path),
                "suite_sha256": sha256(suite_path),
                "wall_elapsed_s": time.time() - wall_start,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "tasks": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[saved] {summary}")


if __name__ == "__main__":
    main()
