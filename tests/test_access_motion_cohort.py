import json
from pathlib import Path
from scripts.run_native_duration_wave import build_wave2_cases,serial_task_groups,case_lane_indices


def test_frozen_stage_has_complete_task_groups_on_two_balanced_pairs(tmp_path):
    spec=json.loads((Path(__file__).resolve().parents[1]/'configs/system/wave2_scenarios.json').read_text())
    rows=build_wave2_cases(spec,'access_motion_first',tmp_path,tmp_path,tmp_path,20261010)
    assert len(rows)==16 and len({r['id'] for r in rows})==16
    assert {r['cohort_pair'] for r in rows}=={0,1}
    lanes=case_lane_indices(rows,2)
    assert [len(x) for x in lanes]==[8,8]
    assert all(len({r['cohort_pair'] for r in rows if r['scenario']==task})==1 for task in {r['scenario'] for r in rows})
    for row in serial_task_groups(rows):
        if row['method']=='no_copy':
            reference=serial_task_groups(rows)[row['reference_case_index']]
            assert reference['scenario']==row['scenario'] and reference['method']=='keep_copy'
    assert all('sum_old' not in r['id'] for r in rows)


def test_factorial_has_no_free_raw_archive_in_restore_off(tmp_path):
    spec=json.loads((Path(__file__).resolve().parents[1]/'configs/system/wave2_scenarios.json').read_text())
    rows=build_wave2_cases(spec,'access_factorial',tmp_path,tmp_path,tmp_path,20261010)
    assert len(rows)==10
    for r in rows:
        if r['method']=='native':assert '--scene-access-mode' not in r['cmd']
        else:assert r['cmd'][r['cmd'].index('--scene-access-mode')+1]==r['method']
