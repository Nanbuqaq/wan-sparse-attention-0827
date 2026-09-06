import copy
import pytest
from scripts.build_aligned_final_probe import build as base_build
from scripts.build_matched_trajectory_capture import build
from scripts.audit_matched_trajectory_capture import compare,route_sequence


def test_capture_preserves_generation_identity_and_does_not_mutate_reference():
    suite,expected=base_build('a'*40)
    original=copy.deepcopy(suite)
    states={'cases':[{**r,'status':'pass'} for r in expected['cases']]}
    capture,same,protocol=build(suite,expected,states,orchestrator_commit='b'*40)
    assert suite==original and same==expected
    assert capture['experiment_commit']=='a'*40
    assert protocol['orchestrator_commit']=='b'*40
    assert all(c['complete_capture'] for c in capture['cases'])
    assert protocol['reference_cases']==6 and protocol['expected_captures_per_case']==40
    states['cases'][0]['status']='fail'
    with pytest.raises(ValueError,match='passed'):build(suite,expected,states,orchestrator_commit='b'*40)


def test_trajectory_gate_checks_noise_and_order_not_only_route_set():
    ref={'initial_noise_sha256':'noise1','case_key_sha256':'same'}
    assert not compare(ref,{**ref,'initial_noise_sha256':'noise2'})['same_initial_noise']
    row={'layer_id':0,'current_start':10,'denoising_pass':0,'route_plan_sha256':'a'}
    a={'call_records':[row,{**row,'denoising_pass':1}]}
    b={'call_records':list(reversed(a['call_records']))}
    assert route_sequence(a)!=route_sequence(b)
