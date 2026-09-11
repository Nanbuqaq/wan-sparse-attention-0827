from types import SimpleNamespace

import pytest

from scripts.run_longlive2_native_reference import validate_source_teacher_protocol


def args(**changes):
    values=dict(capture_attention_teacher=True,causal_block_policy='full',causal_block_fraction=1.,
        causal_block_grouping='flat64',causal_block_heads='shared',causal_block_normalization='source_only',
        causal_block_refresh='first_only',causal_block_query_reduction='mean',equivalence_reference='existing.json')
    values.update(changes)
    return SimpleNamespace(**values)


def test_only_full_source_observation_with_existing_reference_is_admitted():
    validate_source_teacher_protocol(args())
    for changed in [dict(causal_block_policy='mass_value'),dict(causal_block_policy='source_mask'),
                    dict(causal_block_fraction=.25),dict(equivalence_reference=None),
                    dict(causal_block_grouping='spacetime2x4'),dict(causal_block_refresh='phase2')]:
        with pytest.raises(ValueError,match='full raw source'):
            validate_source_teacher_protocol(args(**changed))


def test_other_qualified_capture_paths_keep_their_existing_validation():
    validate_source_teacher_protocol(args(causal_block_policy=None))
    validate_source_teacher_protocol(args(capture_attention_teacher=False,causal_block_policy='mass_value'))


def test_full_query_capture_does_not_widen_online_or_other_capture_paths():
    validate_source_teacher_protocol(args(attention_teacher_query_mode='full'))
    for changed in [dict(capture_attention_teacher=False),dict(causal_block_policy=None),
                    dict(causal_block_policy='mass_value'),dict(equivalence_reference=None)]:
        with pytest.raises(ValueError):
            validate_source_teacher_protocol(args(attention_teacher_query_mode='full',**changed))
