import pytest
from adapters.longlive_sparse.resident_snapshot import oldest_same_phase_window


def test_oldest_snapshot_uses_actual_retained_phase_not_global_or_fictional_initial_time():
    owners=[('native',i,5,0.) for i in range(8)]+[('native',i,20,8.) for i in range(24,48)]
    frames,slots=oldest_same_phase_window(owners,8.)
    assert frames==list(range(24,32)) and slots==list(range(8,16))
    with pytest.raises(ValueError):oldest_same_phase_window(owners,24.)


def test_discontiguous_snapshot_is_rejected_without_latest_fallback():
    owners=[('native',i,5,8.) for i in [8,9,10,11,13,14,15,16]]
    with pytest.raises(ValueError):oldest_same_phase_window(owners,8.)


def test_state_snapshot_cohort_holds_reader_context_budget_and_requests_fixed(tmp_path):
    import json
    from pathlib import Path
    from scripts.run_native_duration_wave import build_wave2_cases,case_lane_indices
    root=Path(__file__).resolve().parents[1];spec=json.loads((root/'configs/system/wave2_scenarios.json').read_text())
    rows=build_wave2_cases(spec,'state_snapshot',tmp_path,tmp_path,tmp_path,20261021)
    assert len(rows)==8 and list(map(len,case_lane_indices(rows,2)))==[4,4]
    assert {x['method'] for x in rows}=={'native','anchor_no_source','latest8','oldest_resident8'}
