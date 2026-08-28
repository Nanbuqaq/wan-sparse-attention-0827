from __future__ import annotations

from scripts.audit_method_smoke_stage import audit_cases


def test_stage_case_audit_rejects_missing_or_failed_cases_without_artifacts():
    expected = [{"id": "a"}, {"id": "b"}]
    records, errors = audit_cases(
        expected,
        [{"cases": [{"id": "a", "status": "fail", "failure_reason": "boom"}]}],
        verify_artifacts=False,
    )
    assert len(records) == 1
    assert any("missing case: b" in error for error in errors)
    assert any("method smoke is not pass: a" in error for error in errors)


def test_stage_case_audit_accepts_complete_pass_states_without_artifacts():
    expected = [{"id": "a"}, {"id": "b"}]
    state = lambda case_id: {
        "id": case_id,
        "status": "pass",
        "backend": "grouped_fa2",
        "stats": "stats.json",
        "config": "config.json",
        "route_plan_sha256": "0" * 64,
        "failed_calls": 0,
        "fallback_calls": 0,
        "nan_calls": 0,
    }
    records, errors = audit_cases(
        expected,
        [{"cases": [state("a")]}, {"cases": [state("b")]}],
        verify_artifacts=False,
    )
    assert len(records) == 2
    assert errors == []
