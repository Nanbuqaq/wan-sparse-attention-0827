import json
from pathlib import Path
from scripts.run_native_duration_wave import build_wave2_cases,case_lane_indices


def test_long_keeps_four_methods_each_existing_seed_on_one_pair(tmp_path):
    spec=json.loads((Path(__file__).resolve().parents[1]/'configs/system/wave2_scenarios.json').read_text())
    rows=build_wave2_cases(spec,'motion_long',tmp_path,tmp_path,tmp_path,20261010)
    lanes=case_lane_indices(rows,2);assert list(map(len,lanes))==[4,4]
    for lane in lanes:
        assert {rows[i]['method'] for i in lane}=={'native','sum_fast','recent','bridge'}
        assert len({rows[i]['cmd'][rows[i]['cmd'].index('--seed')+1] for i in lane})==1
        assert all(rows[i]['latent_frames']==728 for i in lane)
    assert {r['cmd'][r['cmd'].index('--seed')+1] for r in rows}=={'20261010','20261011'}


def test_archive_timing_reverses_order_without_algorithms(tmp_path):
    spec=json.loads((Path(__file__).resolve().parents[1]/'configs/system/wave2_scenarios.json').read_text())
    rows=build_wave2_cases(spec,'archive_timing',tmp_path,tmp_path,tmp_path,20261010)
    for lane in case_lane_indices(rows,2):
        assert [rows[i]['method'] for i in lane]==['no_copy_r0','keep_copy_r0','keep_copy_r1','no_copy_r1']


def test_weight_has_partial_backend_control_and_return_has_native(tmp_path):
    spec=json.loads((Path(__file__).resolve().parents[1]/'configs/system/wave2_scenarios.json').read_text())
    rows=build_wave2_cases(spec,'source_weight',tmp_path,tmp_path,tmp_path,20261010)
    lanes=case_lane_indices(rows,2);assert list(map(len,lanes))==[5,5]
    assert {rows[i]['method'] for i in lanes[0]}=={'native','restore_broad','beta1','beta_half','beta2'}
    assert all('--source-weight-replay' not in r['cmd'] for r in rows)
    delayed=build_wave2_cases(spec,'delayed_and_return',tmp_path,tmp_path,tmp_path,20260913)
    lanes=case_lane_indices(delayed,2);assert list(map(len,lanes))==[5,4]
    assert {delayed[i]['method'] for i in lanes[1]}=={'native','recent','early_heavy','late_heavy'}


def test_context_wave_reuses_prior_native_references_and_adds_room_control(tmp_path):
    spec=json.loads((Path(__file__).resolve().parents[1]/'configs/system/wave2_scenarios.json').read_text())
    rows=build_wave2_cases(spec,'context_controls',tmp_path,tmp_path,tmp_path,20261010)
    assert len(rows)==14 and [len(x) for x in case_lane_indices(rows,2)]==[8,6]
    assert sum('--same-subject-new-room' in r['cmd'] for r in rows)==2
    assert len({r['id'] for r in rows})==14
