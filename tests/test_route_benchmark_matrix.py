from __future__ import annotations

from scripts.run_route_benchmark_matrix import SNAPSHOTS, classify


def test_route_snapshot_names_are_frozen_early_middle_late():
    assert list(SNAPSHOTS) == ["early", "middle", "late"]
    assert SNAPSHOTS["early"].endswith("00028080.pt")
    assert SNAPSHOTS["middle"].endswith("00093600.pt")
    assert SNAPSHOTS["late"].endswith("00177840.pt")


def test_optional_kernel_failure_is_negative_not_grouped_failure():
    payload = {
        "backends": {
            "grouped_fa2": {"status": "pass", "same_route_plan": True},
            "fixed64_rect": {"status": "fail", "same_route_plan": True},
            "varlen_triton": {"status": "pass", "same_route_plan": True},
        }
    }
    status, negatives = classify(payload)
    assert status == "negative"
    assert negatives[0]["backend"] == "fixed64_rect"
