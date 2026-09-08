import pytest

from scripts.run_native_context_study import ARMS,arm_arguments


def test_common_capacity_optimization_is_available_to_every_arm():
    assert all('--cfg1-positive-cache-only' in arm_arguments(a) for a in ARMS)
    assert '--scene-context-reset' not in arm_arguments('none')
    assert '--scene-context-reset' not in arm_arguments('shot')
    assert '--scene-context-reset' in arm_arguments('reset_reveal')
    assert 'raw_away' in arm_arguments('reset_away')


def test_unknown_arm_is_not_silently_converted_to_a_control():
    with pytest.raises(ValueError):arm_arguments('global')
