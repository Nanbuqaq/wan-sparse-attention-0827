from adapters.longlive_sparse.return_context import permitted_native_frames,contiguous_frame_runs


def test_first_transition_controls_match_removed_frames_and_preserve_current_global():
    owners=[('native',0,5,0.),('native',48,35,16.),('native',88,60,16.),('native',96,61,24.)]
    roles=[dict(current=False,pin=False),dict(current=False,pin=True),dict(current=False,pin=False),dict(current=True,pin=False)]
    args=(owners,[0,1,2,3],roles,24.)
    assert permitted_native_frames('pin_first',*args,first_return=True,returning=True,global_slots=1)==[0,2,3]
    assert permitted_native_frames('recent_first',*args,first_return=True,returning=True,global_slots=1)==[0,1,3]
    assert permitted_native_frames('both_first',*args,first_return=True,returning=True,global_slots=1)==[0,3]
    assert permitted_native_frames('pin_first',*args,first_return=False,returning=True,global_slots=1)==[0,1,2,3]
    assert permitted_native_frames('anchor_transition',*args,first_return=False,returning=True,global_slots=1)==[0,3]


def test_context_runs_have_no_padding_or_duplicate_frames():
    assert contiguous_frame_runs([0,1,4,5,6,9])==[(0,2),(4,7),(9,10)]


def test_context_cohort_keeps_legacy_slot_control_and_both_source_factor_arms(tmp_path):
    import json
    from pathlib import Path
    from scripts.run_native_duration_wave import build_wave2_cases,case_lane_indices
    root=Path(__file__).resolve().parents[1];spec=json.loads((root/'configs/system/wave2_scenarios.json').read_text())
    rows=build_wave2_cases(spec,'return_context',tmp_path,tmp_path,tmp_path,20261010)
    assert len(rows)==16 and list(map(len,case_lane_indices(rows,2)))==[8,8]
    legacy=[x for x in rows if x['method']=='legacy_slot']
    assert all('--source-lifetime-policy' not in x['cmd'] for x in legacy)
    assert all('--source-packing-order' in x['cmd'] for x in rows if x['method'] not in ('native','legacy_slot'))
    combined=build_wave2_cases(spec,'context_write',tmp_path,tmp_path,tmp_path,20261010)
    assert len(combined)==24 and list(map(len,case_lane_indices(combined,2)))==[12,12]


def test_current_phase_control_removes_global_and_away_but_retains_current_and_return_history():
    owners=[('native',0,5,0.),('native',48,35,16.),('native',96,65,24.),('native',104,66,24.)]
    roles=[dict(current=False,pin=False),dict(current=False,pin=True),dict(current=False,pin=True),dict(current=True,pin=False)]
    args=(owners,[0,1,2,3],roles,24.)
    assert permitted_native_frames('current_transition',*args,first_return=False,returning=True,global_slots=1)==[2,3]
    assert permitted_native_frames('anchor_transition',*args,first_return=False,returning=True,global_slots=1)==[0,2,3]
    assert permitted_native_frames('current_transition',*args,first_return=False,returning=False,global_slots=1)==[0,1,2,3]
