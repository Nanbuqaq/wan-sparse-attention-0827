from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_four_eight_gpu_partitions_are_complete_disjoint_and_bounded(tmp_path):
    full = tmp_path / "full"
    partitions = tmp_path / "partitions"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_pareto_suites.py"),
            "--frozen-prompts",
            str(ROOT / "configs/formal/frozen_prompts.json"),
            "--selection",
            str(ROOT / "configs/formal/pareto_selection_20260903.json"),
            "--calibration",
            str(ROOT / "configs/formal/method_params.json"),
            "--commit",
            "a" * 40,
            "--output-dir",
            str(full),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_pareto_partition_plan.py"),
            "--expected",
            str(full / "expected_pareto_expansion.json"),
            "--dense-suite",
            str(full / "rag_dense_pareto_expansion.json"),
            "--sparse-suite",
            str(full / "rag_pareto_expansion.json"),
            "--output-dir",
            str(partitions),
            "--partitions",
            "4",
            "--lanes",
            "8",
            "--max-lane-hours",
            "8",
        ],
        check=True,
    )

    expected = json.loads(
        (full / "expected_pareto_expansion.json").read_text(encoding="utf-8")
    )["cases"]
    plan = json.loads(
        (partitions / "partition_plan.json").read_text(encoding="utf-8")
    )
    assert plan["cases"] == 102
    assert plan["partitions"] == 4
    assert plan["lanes_per_partition"] == 8
    assert plan["max_estimated_wall_hours"] < 8

    observed = []
    partition_id_sets = []
    for record in plan["partition_records"]:
        assert len(record["lanes"]) == 8
        assert all(lane["tasks"] for lane in record["lanes"])
        assert record["estimated_wall_hours"] < 8
        assert record["gpu_active_start_lanes"] >= 4
        assert len(set(record["first_methods"])) >= 3
        ids = {
            task["id"]
            for lane in record["lanes"]
            for task in lane["tasks"]
        }
        assert len(ids) == record["case_count"]
        partition_id_sets.append(ids)
        observed.extend(ids)

    for index, left in enumerate(partition_id_sets):
        for right in partition_id_sets[index + 1 :]:
            assert left.isdisjoint(right)
    assert len(observed) == 102
    assert len(set(observed)) == 102
    assert set(observed) == {case["id"] for case in expected}

    task_suites = list(partitions.glob("partition*/lane*_task*.json"))
    assert len(task_suites) == 102
    assert all(
        len(json.loads(path.read_text(encoding="utf-8"))["cases"]) == 1
        for path in task_suites
    )


def test_partition_runner_has_idle_and_resume_guards():
    runner = ROOT / "scripts/inferhub_batch_pareto_partition8.sh"
    subprocess.run(["bash", "-n", str(runner)], check=True)
    source = runner.read_text(encoding="utf-8")
    assert "LONGLIVE_CPU_THREADS_PER_LANE" in source
    assert "reusing existing terminal state" in source
    assert "LONGLIVE_PARETO_PARTITION" in source
    assert "CUDA_VISIBLE_DEVICES=${device}" in source
    assert "CUDA_VISIBLE_DEVICES=0" not in source
