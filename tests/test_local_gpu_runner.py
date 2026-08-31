from __future__ import annotations

from scripts.run_on_free_gpu import eligible_gpu_rows


def test_local_gpu_runner_selects_only_idle_rows_in_stable_order():
    rows = [
        {"index": 1, "memory": 2, "utilization": 0},
        {"index": 0, "memory": 400, "utilization": 1},
        {"index": 2, "memory": 2000, "utilization": 0},
    ]
    assert eligible_gpu_rows(
        rows, max_memory_mib=1024, max_utilization=20
    ) == [rows[0], rows[1]]


def test_parallel_lane_can_filter_an_explicit_physical_gpu():
    rows = eligible_gpu_rows(
        [
            {"index": 0, "memory": 2, "utilization": 0},
            {"index": 1, "memory": 2, "utilization": 0},
        ],
        max_memory_mib=1024,
        max_utilization=20,
    )
    assert [row for row in rows if row["index"] == 1] == [rows[1]]
