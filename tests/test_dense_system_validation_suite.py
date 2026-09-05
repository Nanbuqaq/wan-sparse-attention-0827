import pytest
from scripts.build_dense_system_validation import build


def test_dense_default_still_has_four_distinct_lanes():
    suites, expected = build('a'*40)
    assert list(suites) == [0, 1, 2, 3]
    assert len(expected['cases']) == 4
    assert all(suite['cases'][0]['latent_frames'] == 39 for suite in suites.values())


def test_long_state_profile_is_not_short_motion_calibration():
    suites, expected = build('a'*40, latent_frames=120, prompt_id='calibration_state', lanes=(0, 3))
    assert list(suites) == [0, 3]
    assert len(expected['cases']) == 2
    assert all(suite['cases'][0]['prompt_id'] == 'calibration_state' for suite in suites.values())
    assert all(suite['cases'][0]['latent_frames'] == 120 for suite in suites.values())
    assert suites[3]['cases'][0]['longlive_system']['gpu_union_cache_budget_mib'] == 4096
    with pytest.raises(ValueError):
        build('a'*40, lanes=(0, 0))


def test_final_uses_same_frozen_admission_on_both_systems():
    suites, expected=build('a'*40, latent_frames=120,prompt_id='calibration_state',lanes=(0,3),
                          method='transfer_vaware_hybrid_history')
    assert suites[0]['method_params']==suites[3]['method_params']
    assert suites[0]['history_density']==suites[3]['history_density']==.25
    params=suites[0]['method_params']['transfer_vaware_hybrid_history']
    assert (params['base_fraction'],params['local_fraction'],params['transfer_multiplier'])==(.7,.15,1.)
    assert suites[3]['cases'][0]['longlive_system']['gpu_union_cache_budget_mib']==768
    assert all(case['routing_stage']=='pre-transfer' for case in expected['cases'])


def test_bounded_archive_option_is_explicit_in_identity():
    suites,expected=build('a'*40,lanes=(3,),archive_offload='pooled_pageable',host_pinned_budget_mib=128)
    system=expected['cases'][0]['case_key']['system']
    assert system['archive_offload']=='pooled_pageable' and system['host_pinned_budget_mib']==128
