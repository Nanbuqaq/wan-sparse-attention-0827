#!/usr/bin/env python3
"""Split the frozen Pareto expansion into balanced eight-GPU sub-eight-hour jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


METHOD_SECONDS_120 = {
    # Conservative priors based on the slow side of the completed 477-frame
    # basic matrix, with room for a fresh model load in every exact-task
    # process.  The plan is a scheduling guard, not a reported speed result.
    "rag_dense": 10000.0,
    "fixed_k256_history": 700.0,
    "scope_ar": 4800.0,
    "transfer_vaware_hybrid_history": 3400.0,
}
LONG_MULTIPLIER = 2.5
METHOD_ORDER = [
    "rag_dense",
    "fixed_k256_history",
    "scope_ar",
    "transfer_vaware_hybrid_history",
]
GPU_ACTIVE_START_METHODS = {"rag_dense", "fixed_k256_history"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_token(case: dict, *, dense: bool) -> tuple:
    base = (case["prompt_id"], int(case["seed"]), int(case["latent_frames"]))
    if dense:
        return base
    return base + (
        float(case["history_density"]),
        case["refresh_policy"],
        case["rope_policy"],
    )


def estimate(case: dict) -> float:
    value = METHOD_SECONDS_120[case["method"]]
    if int(case["latent_frames"]) == 240:
        value *= LONG_MULTIPLIER
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", required=True)
    parser.add_argument("--dense-suite", required=True)
    parser.add_argument("--sparse-suite", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument("--lanes", type=int, default=8)
    parser.add_argument("--max-lane-hours", type=float, default=8.0)
    args = parser.parse_args()
    expected_path = Path(args.expected).resolve()
    dense_path = Path(args.dense_suite).resolve()
    sparse_path = Path(args.sparse_suite).resolve()
    expected = json.loads(expected_path.read_text(encoding="utf-8"))["cases"]
    dense_suite = json.loads(dense_path.read_text(encoding="utf-8"))
    sparse_suite = json.loads(sparse_path.read_text(encoding="utf-8"))
    dense_cases = {case_token(case, dense=True): case for case in dense_suite["cases"]}
    sparse_cases = {case_token(case, dense=False): case for case in sparse_suite["cases"]}
    slots = args.partitions * args.lanes
    expected_ids = [case["id"] for case in expected]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("expected case ids must be unique")
    if len(expected) < slots:
        raise ValueError("partition plan requires at least one task per GPU slot")

    tasks = []
    for case in expected:
        method = case["method"]
        if method not in METHOD_SECONDS_120:
            raise ValueError(f"missing runtime prior: {method}")
        dense = method == "rag_dense"
        source = (dense_cases if dense else sparse_cases).get(case_token(case, dense=dense))
        if source is None:
            raise RuntimeError(f"case config not found for {case['id']}")
        tasks.append(
            {
                "id": case["id"],
                "case_key_sha256": case["case_key_sha256"],
                "method": method,
                "kind": "dense" if dense else "sparse",
                "case": dict(source),
                "expected": dict(case),
                "estimated_seconds": estimate(case),
            }
        )

    lane_tasks = [[] for _ in range(slots)]
    lane_loads = [0.0 for _ in range(slots)]
    for task in sorted(tasks, key=lambda item: (-item["estimated_seconds"], item["id"])):
        slot = min(range(slots), key=lambda index: (lane_loads[index], index))
        lane_tasks[slot].append(task)
        lane_loads[slot] += task["estimated_seconds"]
    if any(not items for items in lane_tasks):
        raise RuntimeError("empty GPU lane after assignment")
    if max(lane_loads) > args.max_lane_hours * 3600:
        raise RuntimeError(
            f"estimated lane {max(lane_loads) / 3600:.2f}h exceeds limit"
        )

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    partitions = []
    for partition in range(args.partitions):
        partition_dir = output / f"partition{partition}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        partial_expected = []
        lanes = []
        first_methods = []
        for lane in range(args.lanes):
            slot = partition * args.lanes + lane
            rotation = lane % len(METHOD_ORDER)
            order = METHOD_ORDER[rotation:] + METHOD_ORDER[:rotation]
            priority = {method: index for index, method in enumerate(order)}
            items = sorted(
                lane_tasks[slot],
                key=lambda item: (
                    priority[item["method"]],
                    -item["estimated_seconds"],
                    item["id"],
                ),
            )
            first_methods.append(items[0]["method"])
            task_records = []
            for task_index, task in enumerate(items):
                suite_name = f"lane{lane}_task{task_index}.json"
                if task["kind"] == "dense":
                    suite = {
                        "status": "frozen_pareto_partition",
                        "experiment_commit": task["expected"]["commit"],
                        "cases": [task["case"]],
                    }
                else:
                    suite = {
                        "status": "frozen_pareto_partition",
                        "experiment_commit": task["expected"]["commit"],
                        "history_density": float(task["case"]["history_density"]),
                        "backend": task["expected"]["backend"],
                        "refresh_policy": task["case"]["refresh_policy"],
                        "rope_policy": task["case"]["rope_policy"],
                        "record_per_call": False,
                        "methods": [task["method"]],
                        "method_params": sparse_suite.get("method_params", {}),
                        "latent_frames": int(task["case"]["latent_frames"]),
                        "cases": [task["case"]],
                    }
                (partition_dir / suite_name).write_text(
                    json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                task_records.append(
                    {
                        "id": task["id"],
                        "kind": task["kind"],
                        "method": task["method"],
                        "suite": suite_name,
                        "estimated_seconds": task["estimated_seconds"],
                    }
                )
                partial_expected.append(task["expected"])
            lane_plan = {
                "lane": lane,
                "estimated_seconds": lane_loads[slot],
                "tasks": task_records,
            }
            lane_name = f"lane{lane}_plan.json"
            (partition_dir / lane_name).write_text(
                json.dumps(lane_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            lanes.append(lane_plan)
        if len(set(first_methods)) < 3:
            raise RuntimeError(
                f"partition {partition} startup methods are insufficiently staggered: {first_methods}"
            )
        gpu_active_start_lanes = sum(
            method in GPU_ACTIVE_START_METHODS for method in first_methods
        )
        minimum_gpu_active_start_lanes = max(1, args.lanes // 2)
        if gpu_active_start_lanes < minimum_gpu_active_start_lanes:
            raise RuntimeError(
                f"partition {partition} has only {gpu_active_start_lanes} "
                "GPU-active startup lanes; synchronized CPU-heavy routing may "
                f"trigger infer_gpu_idle: {first_methods}"
            )
        expected_name = "expected.json"
        (partition_dir / expected_name).write_text(
            json.dumps({"cases": sorted(partial_expected, key=lambda case: case["id"])}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        partition_record = {
            "partition": partition,
            "expected": str((partition_dir / expected_name).relative_to(output)),
            "case_count": len(partial_expected),
            "estimated_gpu_hours": sum(lane["estimated_seconds"] for lane in lanes) / 3600,
            "estimated_wall_hours": max(lane["estimated_seconds"] for lane in lanes) / 3600,
            "first_methods": first_methods,
            "gpu_active_start_lanes": gpu_active_start_lanes,
            "minimum_gpu_active_start_lanes": minimum_gpu_active_start_lanes,
            "lanes": lanes,
        }
        (partition_dir / "partition_plan.json").write_text(
            json.dumps(partition_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        partitions.append(partition_record)

    observed = sorted(task["id"] for items in lane_tasks for task in items)
    wanted = sorted(case["id"] for case in expected)
    if observed != wanted:
        raise RuntimeError("partition union does not equal expected cases")
    plan = {
        "status": "frozen_pareto_partition_plan",
        "source": {
            "expected": {"path": str(expected_path), "sha256": sha256(expected_path)},
            "dense_suite": {"path": str(dense_path), "sha256": sha256(dense_path)},
            "sparse_suite": {"path": str(sparse_path), "sha256": sha256(sparse_path)},
        },
        "partitions": args.partitions,
        "lanes_per_partition": args.lanes,
        "cases": len(expected),
        "max_estimated_wall_hours": max(item["estimated_wall_hours"] for item in partitions),
        "runtime_priors_seconds_120": METHOD_SECONDS_120,
        "long_multiplier": LONG_MULTIPLIER,
        "gpu_active_start_methods": sorted(GPU_ACTIVE_START_METHODS),
        "partition_records": partitions,
    }
    (output / "partition_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": plan["status"],
                "cases": plan["cases"],
                "partitions": plan["partitions"],
                "case_counts": [item["case_count"] for item in partitions],
                "estimated_wall_hours": [round(item["estimated_wall_hours"], 3) for item in partitions],
                "max_estimated_wall_hours": round(plan["max_estimated_wall_hours"], 3),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
