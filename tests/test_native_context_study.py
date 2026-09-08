import pytest

from scripts.run_native_context_study import ARMS,arm_arguments
from scripts.review_native_memory_study import native_review_indices


def test_common_capacity_optimization_is_available_to_every_arm():
    assert all('--cfg1-positive-cache-only' in arm_arguments(a) for a in ARMS)
    assert '--scene-context-reset' not in arm_arguments('none')
    assert '--scene-context-reset' not in arm_arguments('shot')
    assert '--scene-context-reset' in arm_arguments('reset_reveal')
    assert 'raw_away' in arm_arguments('reset_away')


def test_unknown_arm_is_not_silently_converted_to_a_control():
    with pytest.raises(ValueError):arm_arguments('global')


def test_review_source_frame_never_reads_first_away_frame():
    ends,comparison=native_review_indices()
    assert ends==(188,380,412,508)
    assert max(comparison[:2])<189
    assert all(253<=f<381 for f in comparison[2:4])
    assert all(381<=f<413 for f in comparison[4:6])
