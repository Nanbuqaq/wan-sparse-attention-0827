from __future__ import annotations

from adapters.longlive_sparse.prefetch import build_verified_prefetch_plan


def test_prefetch_caps_churn_and_completes_exact_actual_route() -> None:
    plan = build_verified_prefetch_plan(
        predicted=[1, 2, 8, 9],
        actual=[0, 1, 2, 3],
        resident=[0],
        bytes_per_block=100,
        max_new_blocks=2,
        max_new_bytes=250,
        ready_before_use=[1],
    )
    assert plan.admitted_predictions == (1, 2)
    assert plan.resident_hits == (0,)
    assert plan.prediction_hits == (1, 2)
    assert plan.completion_misses == (3,)
    assert plan.extras == ()
    assert plan.newly_admitted_bytes == 200
    assert plan.miss_bytes == 100
    assert plan.timeliness == 0.5
    assert plan.final_execution_blocks() == (0, 1, 2, 3)


def test_prefetch_reports_extra_bytes_without_changing_execution() -> None:
    plan = build_verified_prefetch_plan(
        predicted=[4, 5, 6],
        actual=[4, 7],
        resident=[],
        bytes_per_block=64,
        max_new_blocks=3,
        max_new_bytes=3 * 64,
        ready_before_use=[4, 5, 6],
    )
    assert plan.extras == (5, 6)
    assert plan.extra_bytes == 128
    assert plan.completion_misses == (7,)
    assert plan.final_execution_blocks() == (4, 7)


def test_resident_predictions_do_not_consume_new_admission_budget() -> None:
    plan = build_verified_prefetch_plan(
        predicted=[1, 2, 3],
        actual=[1, 2, 3],
        resident=[1],
        bytes_per_block=10,
        max_new_blocks=2,
        max_new_bytes=20,
    )
    assert plan.admitted_predictions == (2, 3)
    assert plan.resident_hits == (1,)
    assert plan.completion_misses == ()
