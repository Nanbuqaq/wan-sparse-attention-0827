from __future__ import annotations

from scripts.build_case_metrics import negative_reasons


def review(**updates):
    values = {
        "irreversible_state_reset_count": 0,
        "action_loop_count": 0,
        "action_discontinuity_count": 0,
        "freeze_count": 0,
        "flicker_count": 0,
        "camera_cut_count": 0,
        "subject_identity_1to5": 5,
        "background_consistency_1to5": 5,
        "late_quarter_quality_1to5": 5,
        "late_quarter_degradation_0to2": 0,
    }
    values.update(updates)
    return values


def case(method, **updates):
    values = {
        "id": method,
        "method": method,
        "status": "pass",
        "manual_review": review(),
        "end_to_end_s": 10.0,
        "attention_s": 4.0,
        "h2d_s": 2.0,
    }
    values.update(updates)
    return values


def test_new_flicker_marks_technical_success_negative():
    baseline = case("rag_dense")
    candidate = case("block64_history", manual_review=review(flicker_count=1))
    assert "new_flicker_count" in negative_reasons(candidate, baseline, finalize=True)


def test_slow_without_component_gain_marks_negative():
    baseline = case("rag_dense")
    candidate = case(
        "block64_history", end_to_end_s=11.0, attention_s=3.9, h2d_s=1.9
    )
    assert "slower_than_dense_without_attention_or_h2d_gain" in negative_reasons(
        candidate, baseline, finalize=True
    )


def test_attention_gain_prevents_speed_only_negative():
    baseline = case("rag_dense")
    candidate = case(
        "block64_history", end_to_end_s=11.0, attention_s=3.5, h2d_s=2.0
    )
    assert negative_reasons(candidate, baseline, finalize=True) == []
