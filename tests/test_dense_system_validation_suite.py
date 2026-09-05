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
