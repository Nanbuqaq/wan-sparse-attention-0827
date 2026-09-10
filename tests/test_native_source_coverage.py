import json
from pathlib import Path

import pytest
import torch

from adapters.longlive_sparse.native_causal_block_memory import (
    CausalBlockConfig, groups_for_source, reduce_query_scores)
from scripts.run_native_causal_block_wave import build_cases


def test_query_peak_preserves_a_low_scale_specialized_query():
    score = torch.tensor([[[100., 1., 0.], [100., 1., 0.], [0., 0., 1.]]])
    assert reduce_query_scores(score,'mean').argmax().item() == 0
    assert reduce_query_scores(score,'normalized_peak').argmax().item() == 2
    scaled = score*torch.tensor([[[.01],[3.],[800.]]])
    torch.testing.assert_close(reduce_query_scores(score,'normalized_peak'),
        reduce_query_scores(scaled,'normalized_peak'))
    assert not reduce_query_scores(torch.zeros_like(score),'normalized_peak').any()


@pytest.mark.parametrize('height,width',[(22,40),(8,16)])
def test_temporal_groups_cover_raw_coordinates_and_have_matched_counts(height,width):
    tube = groups_for_source(height,width,kind='spacetime2x4')
    flat = groups_for_source(height,width,kind='flat_tube_matched')
    assert [len(g) for g in tube] == [len(g) for g in flat]
    for groups in (tube,flat):
        assert sorted(t for g in groups for t in g)==list(range(8*height*width))
        assert all(len(g)==64 for g in groups)
    for group in tube:
        assert {t//(height*width) for t in group}==set(range(8))
        sites={t%(height*width) for t in group}
        assert len(sites)==8
        assert len({s//width for s in sites})==2
        assert len({s%width for s in sites})==4


def test_unregistered_factor_combinations_are_rejected():
    with pytest.raises(ValueError):
        CausalBlockConfig(policy='mass_value',fraction=.25,grouping='spacetime2x4',
            head_policy='per_head',query_reduction='normalized_peak')
    with pytest.raises(ValueError):
        CausalBlockConfig(policy='source_mask',fraction=.25,query_reduction='normalized_peak')


def test_new_frozen_wave_forwards_the_registered_controls():
    spec=json.loads((Path(__file__).resolve().parents[1]/'configs/system/native_source_coverage_wave.json').read_text())
    cases=build_cases(spec,'screen',Path('/assets'),Path('/source'),Path('/out'))
    assert len(cases)==len({c['id'] for c in cases})==12
    assert all('--native-shared-conditioning' in c['cmd'] for c in cases)
    for case in cases:
        assert case['cmd'][case['cmd'].index('--seed')+1]=='20261003'
        if case['method']['id']=='spatial_query_peak':
            assert case['cmd'][case['cmd'].index('--causal-block-query-reduction')+1]=='normalized_peak'
