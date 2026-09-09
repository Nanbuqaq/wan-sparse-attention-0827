from collections import Counter

import pytest
import torch

from scripts.analyze_distance_matched_region import distance_key,matched_indices,masked_attention


def test_controls_match_source_frame_and_distance_bin_without_overlap():
    region=[0,1,16,17];eligible=[2,3,18,19]
    result=matched_indices(region,eligible,5,grid=(4,4),frames=2,bucket_squared=16)
    assert result['matched']==4 and not set(result['region'])&set(result['control'])
    key=lambda x:distance_key(x,5,(4,4),16)
    assert Counter(map(key,result['region']))==Counter(map(key,result['control']))
    assert result==matched_indices(region,eligible,5,grid=(4,4),frames=2,bucket_squared=16)


def test_insufficient_controls_reduce_both_sides_and_report_coverage():
    result=matched_indices([0,1,2],[3],5,grid=(4,4),frames=1,bucket_squared=16)
    assert result['matched']==1 and result['coverage']==pytest.approx(1/3)
    assert len(result['region'])==len(result['control'])


def test_region_and_control_are_disjoint_valid_source_coordinates():
    with pytest.raises(ValueError):matched_indices([1],[1],0)
    with pytest.raises(ValueError):matched_indices([-1],[1],0)
    with pytest.raises(ValueError):matched_indices([0,0],[1],0)


def test_query_specific_removal_recomputes_softmax_instead_of_zeroing_output():
    logits=torch.tensor([[[1.,2.,3.],[3.,2.,1.]]])
    values=torch.tensor([[[1.,0.],[0.,1.],[2.,2.]]])
    mask=torch.tensor([[False,True,False],[True,False,False]])
    actual=masked_attention(logits,values,mask)
    for query in range(2):
        kept=~mask[query]
        expected=logits[0,query,kept].softmax(-1)@values[0,kept]
        assert torch.allclose(actual[0,query],expected)
    with pytest.raises(ValueError):masked_attention(logits,values,torch.ones_like(mask))
