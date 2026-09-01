#!/usr/bin/env python3
"""Checkpoint completed formal cases and freeze an eight-lane residual plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.longlive_sparse.case_identity import validate_case_identity


TERMINAL = {"pass", "fail", "negative"}
BASELINES = {"native_dense", "native_block", "rag_dense"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_terminal_states(source_root: Path) -> tuple[list[dict], list[str]]:
    states = {}
    sources = []
    for path in sorted(source_root.rglob("dense_screen_states.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        sources.append(str(path.resolve()))
        for case in payload.get("cases", []):
            case_id = case.get("id", case.get("case_id"))
            if case.get("status") not in TERMINAL or not case_id:
                continue
            if case_id in states and states[case_id] != case:
                raise RuntimeError(f"conflicting checkpoint state: {case_id}")
            states[case_id] = case
    for path in sorted(source_root.rglob("case_state.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        case_id = case.get("id", case.get("case_id"))
        if case.get("status") not in TERMINAL or not case_id:
            continue
        sources.append(str(path.resolve()))
        if case_id in states and states[case_id] != case:
            raise RuntimeError(f"conflicting checkpoint state: {case_id}")
        states[case_id] = case
    return [states[key] for key in sorted(states)], sorted(set(sources))


def task_seconds(
    method: str,
    *,
    completed_by_method: dict[str, list[float]],
    priors: dict,
) -> float:
    observed = completed_by_method.get(method, [])
    if observed:
        return float(statistics.median(observed))
    return float(
        priors.get("method_seconds_per_477_case", {}).get(
            method, priors["default_seconds_per_477_case"]
        )
    )


def assign_tasks(tasks: list[dict], lane_count: int) -> tuple[list[list[dict]], list[float]]:
    if lane_count <= 0 or len(tasks) < lane_count:
        raise ValueError("residual plan requires at least one distinct task per lane")
    lanes = [[] for _ in range(lane_count)]
    loads = [0.0 for _ in range(lane_count)]
    for task in sorted(
        tasks,
        key=lambda item: (-float(item["estimated_seconds"]), item["id"]),
    ):
        lane = min(range(lane_count), key=lambda index: (loads[index], index))
        lanes[lane].append(task)
        loads[lane] += float(task["estimated_seconds"])
    return lanes, loads


def lane_suite_specs(lane_tasks: list[dict], suite: dict) -> list[dict]:
    all_prompt_ids = [case["prompt_id"] for case in suite["cases"]]
    tasks_by_method = defaultdict(dict)
    for task in lane_tasks:
        tasks_by_method[task["method"]][task["prompt_id"]] = task

    specs = []
    both_methods = sorted(
        method
        for method, prompts in tasks_by_method.items()
        if sorted(prompts) == sorted(all_prompt_ids)
    )
    consumed = set()
    if both_methods:
        specs.append(
            {
                "methods": both_methods,
                "cases": [dict(case) for case in suite["cases"]],
                "task_ids": sorted(
                    task["id"]
                    for method in both_methods
                    for task in tasks_by_method[method].values()
                ),
            }
        )
        consumed.update((method, prompt) for method in both_methods for prompt in all_prompt_ids)
    for case in suite["cases"]:
        prompt_id = case["prompt_id"]
        methods = sorted(
            method
            for method, prompts in tasks_by_method.items()
            if prompt_id in prompts and (method, prompt_id) not in consumed
        )
        if methods:
            specs.append(
                {
                    "methods": methods,
                    "cases": [dict(case)],
                    "task_ids": sorted(tasks_by_method[method][prompt_id]["id"] for method in methods),
                }
            )
    observed = sorted(task_id for spec in specs for task_id in spec["task_ids"])
    expected = sorted(task["id"] for task in lane_tasks)
    if observed != expected:
        raise RuntimeError("lane suite Cartesian products do not match assigned tasks")
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-commit", required=True)
    parser.add_argument("--lane-count", type=int, default=8)
    parser.add_argument(
        "--runtime-priors", default="configs/formal/residual_runtime_priors.json"
    )
    args = parser.parse_args()
    source_root = Path(args.source_root).resolve()
    expected_path = Path(args.expected).resolve()
    suite_path = Path(args.suite).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected_payload = json.loads(expected_path.read_text(encoding="utf-8"))
    expected = expected_payload["cases"]
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    priors_path = Path(args.runtime_priors)
    if not priors_path.is_absolute():
        priors_path = ROOT / priors_path
    priors = json.loads(priors_path.read_text(encoding="utf-8"))
    if priors.get("status") != "scheduling_only":
        raise ValueError("runtime priors are not scheduling-only evidence")
    commits = {case.get("commit") for case in expected}
    if commits != {args.experiment_commit}:
        raise ValueError(f"expected manifest commit mismatch: {commits}")
    if suite.get("experiment_commit") != args.experiment_commit:
        raise ValueError("suite experiment commit mismatch")

    completed, state_sources = collect_terminal_states(source_root)
    expected_by_id = {case["id"]: case for case in expected}
    completed_by_id = {}
    for state in completed:
        case_id = state.get("id", state.get("case_id"))
        if case_id not in expected_by_id:
            raise RuntimeError(f"checkpoint state not in expected matrix: {case_id}")
        errors = validate_case_identity(state)
        if errors:
            raise ValueError(f"invalid checkpoint identity {case_id}: {errors}")
        if state.get("case_key_sha256") != expected_by_id[case_id].get("case_key_sha256"):
            raise ValueError(f"checkpoint/expected identity mismatch: {case_id}")
        completed_by_id[case_id] = state
    completed_expected = [case for case in expected if case["id"] in completed_by_id]
    residual_expected = [case for case in expected if case["id"] not in completed_by_id]
    missing_baselines = [case["id"] for case in residual_expected if case["method"] in BASELINES]
    if missing_baselines:
        raise RuntimeError(f"residual RAG runner cannot recover missing baselines: {missing_baselines}")

    checkpoint_states = {
        "sources": [
            {"artifact_id": Path(path).name, "path": path}
            for path in state_sources
        ],
        "cases": [completed_by_id[case["id"]] for case in completed_expected],
    }
    checkpoint_expected_path = output / "checkpoint_expected.json"
    checkpoint_states_path = output / "checkpoint_states.json"
    checkpoint_audit_path = output / "checkpoint_terminal_audit.json"
    checkpoint_expected_path.write_text(
        json.dumps({"cases": completed_expected}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checkpoint_states_path.write_text(
        json.dumps(checkpoint_states, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_case_states.py"),
            "--expected",
            str(checkpoint_expected_path),
            "--states",
            str(checkpoint_states_path),
            "--output",
            str(checkpoint_audit_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (output / "checkpoint_audit.log").write_text(audit.stdout, encoding="utf-8")
    if audit.returncode:
        raise RuntimeError("completed-case checkpoint audit failed")

    cases_by_prompt = {case["prompt_id"]: case for case in suite["cases"]}
    suite_methods = set(suite["methods"])
    completed_timings = defaultdict(list)
    for state in completed_by_id.values():
        elapsed = state.get("end_to_end_s", state.get("elapsed_s"))
        if elapsed is not None:
            completed_timings[state["method"]].append(float(elapsed))
    tasks = []
    for case in residual_expected:
        if case["method"] not in suite_methods or case["prompt_id"] not in cases_by_prompt:
            raise RuntimeError(f"residual case cannot be represented by frozen RAG suite: {case['id']}")
        tasks.append(
            {
                "id": case["id"],
                "case_key_sha256": case["case_key_sha256"],
                "method": case["method"],
                "prompt_id": case["prompt_id"],
                "estimated_seconds": task_seconds(
                    case["method"], completed_by_method=completed_timings, priors=priors
                ),
            }
        )
    lanes, loads = assign_tasks(tasks, args.lane_count)
    lane_plans = []
    for lane_index, lane_tasks in enumerate(lanes):
        specs = lane_suite_specs(lane_tasks, suite)
        suite_records = []
        for suite_index, spec in enumerate(specs):
            suite_name = f"lane{lane_index}_suite{suite_index}.json"
            payload = {
                **{key: value for key, value in suite.items() if key not in {"methods", "cases"}},
                "status": "frozen_basic_residual",
                "residual_source": {
                    "expected_sha256": sha256(expected_path),
                    "suite_sha256": sha256(suite_path),
                    "checkpoint_states_sha256": sha256(checkpoint_states_path),
                },
                "methods": spec["methods"],
                "cases": spec["cases"],
                "expected_task_ids": spec["task_ids"],
            }
            (output / suite_name).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            suite_records.append(
                {
                    "suite": suite_name,
                    "task_ids": spec["task_ids"],
                    "methods": spec["methods"],
                    "prompt_ids": [case["prompt_id"] for case in spec["cases"]],
                }
            )
        lane_plan = {
            "lane_index": lane_index,
            "estimated_seconds": loads[lane_index],
            "tasks": lane_tasks,
            "suites": suite_records,
        }
        lane_name = f"lane{lane_index}_plan.json"
        (output / lane_name).write_text(
            json.dumps(lane_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        lane_plans.append({"plan": lane_name, **lane_plan})

    residual_expected_path = output / "residual_expected.json"
    original_expected_path = output / "expected_basic_477.json"
    residual_expected_path.write_text(
        json.dumps({"cases": residual_expected}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    original_expected_path.write_text(
        json.dumps(expected_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runner_source = ROOT / "scripts" / "inferhub_batch_basic_residual_8gpu.sh"
    runner_copy = output / runner_source.name
    priors_copy = output / priors_path.name
    shutil.copy2(runner_source, runner_copy)
    shutil.copy2(priors_path, priors_copy)
    orchestration_commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    plan = {
        "status": "frozen_basic_residual",
        "experiment_commit": args.experiment_commit,
        "source_root": str(source_root),
        "expected_cases": len(expected),
        "checkpoint_cases": len(completed_expected),
        "residual_cases": len(residual_expected),
        "lane_count": args.lane_count,
        "checkpoint_terminal_audit": {
            "artifact_id": checkpoint_audit_path.name,
            "sha256": sha256(checkpoint_audit_path),
        },
        "orchestration": {
            "commit": orchestration_commit,
            "runner": {"artifact_id": runner_copy.name, "sha256": sha256(runner_copy)},
        },
        "runtime_priors": {"artifact_id": priors_copy.name, "sha256": sha256(priors_copy)},
        "lanes": lane_plans,
    }
    (output / "residual_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "checkpoint_cases": len(completed_expected),
                "residual_cases": len(residual_expected),
                "lane_load_hours": [round(value / 3600, 2) for value in loads],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
