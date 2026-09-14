import pytest
from adapters.longlive_sparse.immutable_source_reader import source_allowed,request_key
from adapters.longlive_sparse.source_lifetime_protocol import lifetime_schedule


def test_full_once_and_prior_three_equal_actual_source_pairs_with_clean():
    exposure={p:sum(source_allowed(p,l,c)*7040*24*7040 for c in range(4) for l in range(30) for call in range(5))
              for p in ('off','full_once','prior_once','full_three','prior_three')}
    assert exposure['off']==0
    assert exposure['full_once']==exposure['prior_three']==178421760000
    assert exposure['full_three']==3*exposure['full_once']
    assert exposure['prior_once']*3==exposure['full_once']
    assert not any(source_allowed(p,l,3) for p in exposure for l in range(30))


def test_request_key_ignores_only_native_cut_prefix_and_changes_with_actual_request():
    assert request_key('The scene transitions. Back to the same jug.')==request_key('Back to the same jug.')
    assert request_key('Back to the same jug.')!=request_key('Back to the empty shelf.')
    with pytest.raises(ValueError):source_allowed('full_once',0,-1)


def test_gate_extends_return_without_moving_source_or_away_or_repeating_cut():
    original=[dict(start_latent=t,prompt=str(t)) for t in (0,8,16,48)]
    prompts=[['a','b','c','c','c','c','The scene transitions. return','return']]
    changed,expanded=lifetime_schedule(original,prompts,gate=True)
    assert [x['start_latent'] for x in changed]==[0,8,16,48]
    assert expanded[0][:8]==prompts[0] and expanded[0][8:]==['return','return']
    assert len(prompts[0])==8


def test_lifetime_cohort_has_same_motion_request_for_all_five_task_arms(tmp_path):
    import json
    from pathlib import Path
    from scripts.run_native_duration_wave import build_wave2_cases,case_lane_indices
    spec=json.loads((Path(__file__).resolve().parents[1]/'configs/system/wave2_scenarios.json').read_text())
    rows=build_wave2_cases(spec,'source_lifetime',tmp_path,tmp_path,tmp_path,20261010)
    lanes=case_lane_indices(rows,2)
    assert len(rows)==10 and list(map(len,lanes))==[5,5]
    for lane,indices in enumerate(lanes):
        assert all(('--source-lifetime-motion' in rows[i]['cmd'])==(lane==1) for i in indices)
        assert {rows[i]['method'] for i in indices}=={'native','full_once','prior_once','full_three','prior_three'}
        assert all('--scene-access-mode' not in rows[i]['cmd'] for i in indices)
