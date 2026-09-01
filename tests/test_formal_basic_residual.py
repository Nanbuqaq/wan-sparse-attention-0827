from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.build_formal_basic_residual import assign_tasks, lane_suite_specs


def test_residual_tasks_are_greedily_balanced_and_unique():
    tasks = [
        {"id": f"t{index}", "estimated_seconds": value, "method": f"m{index}", "prompt_id": "p0"}
        for index, value in enumerate((10, 9, 8, 7, 6, 5, 4, 3, 2, 1))
    ]
    lanes, loads = assign_tasks(tasks, 4)
    assert sorted(task["id"] for lane in lanes for task in lane) == sorted(task["id"] for task in tasks)
    assert all(lanes)
    assert max(loads) - min(loads) <= 3


def test_lane_suites_match_exact_task_cartesian_products():
    suite = {
        "cases": [
            {"prompt_id": "p0", "prompt": "zero", "seed": 1},
            {"prompt_id": "p1", "prompt": "one", "seed": 1},
        ]
    }
    tasks = [
        {"id": "m0-p0", "method": "m0", "prompt_id": "p0"},
        {"id": "m0-p1", "method": "m0", "prompt_id": "p1"},
        {"id": "m1-p1", "method": "m1", "prompt_id": "p1"},
    ]
    specs = lane_suite_specs(tasks, suite)
    assert specs[0]["methods"] == ["m0"]
    assert [case["prompt_id"] for case in specs[0]["cases"]] == ["p0", "p1"]
    assert specs[1]["methods"] == ["m1"]
    assert [case["prompt_id"] for case in specs[1]["cases"]] == ["p1"]
    assert sorted(task for spec in specs for task in spec["task_ids"]) == ["m0-p0", "m0-p1", "m1-p1"]


def test_lane_suites_start_with_gpu_active_real_tasks():
    suite = {
        "cases": [
            {"prompt_id": "p0", "prompt": "zero", "seed": 1},
            {"prompt_id": "p1", "prompt": "one", "seed": 1},
        ]
    }
    tasks = [
        {"id": "coverage", "method": "coverage_cluster_history", "prompt_id": "p0"},
        {"id": "scope", "method": "scope_ar", "prompt_id": "p0"},
        {"id": "final", "method": "transfer_vaware_hybrid_history", "prompt_id": "p1"},
    ]
    specs = lane_suite_specs(tasks, suite)
    assert specs[0]["methods"] == ["scope_ar", "coverage_cluster_history"]
    assert specs[1]["methods"] == ["transfer_vaware_hybrid_history"]


def test_eight_gpu_residual_runner_is_shell_valid():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/inferhub_batch_basic_residual_8gpu.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text(encoding="utf-8")
    assert "requires exactly eight assigned GPUs" in text
    assert "checkpoint_states.json" in text
    assert "--shard-index 0 --shard-count 1" in text
    assert "terminal_state_audit.json" in text


def test_four_gpu_partition_runner_is_shell_valid():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/inferhub_batch_basic_residual_partition4.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text(encoding="utf-8")
    assert "requires exactly four assigned GPUs" in text
    assert "partial_merged_case_states.json" in text
    assert "partial_terminal_state_audit.json" in text


def test_full_residual_runner_is_shell_valid():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/inferhub_batch_basic_residual_full.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text(encoding="utf-8")
    assert "plan requires" in text
    assert "checkpoint_states.json" in text
    assert "merged_case_states.json" in text
    assert "terminal_state_audit.json" in text
