import pytest
from adapters.longlive_sparse.immutable_source_reader import source_stage_allows


def test_matched_four_call_controls_include_or_exclude_clean_explicitly():
    actual={p:[s for s in range(5) if source_stage_allows(p,s)] for p in ['all','no_clean','no_first','no_last']}
    assert actual==dict(all=[0,1,2,3,4],no_clean=[0,1,2,3],no_first=[1,2,3,4],no_last=[0,1,2,4])
    assert all(len(actual[p])==4 for p in ['no_clean','no_first','no_last'])
    with pytest.raises(ValueError):source_stage_allows('all',5)


def test_single_source_call_positions_are_explicit_and_equal_budget():
    for policy,index in [('first_only',0),('second_only',1),('last_only',3),('clean_only',4)]:
        assert [s for s in range(5) if source_stage_allows(policy,s)]==[index]


def test_four_contexts_keep_prefix_reference_on_the_same_lane(tmp_path):
    import json
    from pathlib import Path
    from scripts.run_native_duration_wave import build_wave2_cases,case_lane_indices
    root=Path(__file__).resolve().parents[1]
    spec=json.loads((root/'configs/system/wave2_scenarios.json').read_text())
    rows=build_wave2_cases(spec,'source_clean',tmp_path,tmp_path,tmp_path,20261023)
    assert len({r['id'] for r in rows})==16
    assert list(map(len,case_lane_indices(rows,4)))==[4,4,4,4]
    for i,r in enumerate(rows):
        if r['method']=='no_clean':
            j=r['reference_case_index']
            assert j<i and rows[j]['cohort_pair']==r['cohort_pair'] and rows[j]['method']=='all'
            assert rows[j]['source_window']==r['source_window']


def test_first_source_cohort_is_two_complete_equal_position_groups(tmp_path):
    import json
    from pathlib import Path
    from scripts.run_native_duration_wave import build_wave2_cases,case_lane_indices
    root=Path(__file__).resolve().parents[1]
    spec=json.loads((root/'configs/system/wave2_scenarios.json').read_text())
    rows=build_wave2_cases(spec,'source_first',tmp_path,tmp_path,tmp_path,20261023)
    assert len({r['id'] for r in rows})==10
    assert list(map(len,case_lane_indices(rows,2)))==[5,5]
    for lane in case_lane_indices(rows,2):
        assert {rows[i]['method'] for i in lane}=={'all','first_only','second_only','last_only','clean_only'}
        assert {rows[i]['source_window'] for i in lane}=={'latest8'}
