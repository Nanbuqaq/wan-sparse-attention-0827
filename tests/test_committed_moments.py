import pytest
import torch
from adapters.longlive_sparse.committed_moments import build_frame_moments, gather_moment_payload, subtract_admitted_raw
from adapters.longlive_sparse.prototype_tail import build_prototype_tail
from adapters.longlive_sparse.feature_prototypes import build_feature_tail


@pytest.mark.parametrize('grouping,groups',[('spatial',1),('key_kmeans',2)])
@pytest.mark.parametrize('selected', [[], [0,3,6,7,11], list(range(14))])
def test_summary_subtraction_matches_full_materialization(grouping, groups, selected):
    gen = torch.Generator().manual_seed(518)
    k = torch.randn(2,14,3,8,generator=gen)
    v = torch.randn(k.shape,generator=gen)
    indices = torch.tensor(selected,dtype=torch.long)[None,None].expand(2,3,-1)
    frames = [build_frame_moments(k[:,i:i+7],v[:,i:i+7],block_tokens=4,grouping=grouping,groups=groups).cpu() for i in (0,7)]
    frame_ids = [17,5]  # non-sorted retrieval order and non-consecutive IDs
    selected_frames = torch.where(indices < 7,17,5)
    payload = gather_moment_payload(frames,frame_ids,selected_frames,indices%7)
    def gather(t):
        return t.permute(0,2,1,3).gather(2,indices[...,None].expand(-1,-1,-1,8)).permute(0,2,1,3)
    actual = subtract_admitted_raw(payload,gather(k),gather(v),frame_tokens=7,block_tokens=4,prototype_dtype=torch.float32)
    if grouping == 'spatial':
        expected = build_prototype_tail(k,v,indices,frame_tokens=7,block_tokens=4,prototype_dtype=torch.float32)
    else:
        expected,_ = build_feature_tail(k,v,indices,frame_tokens=7,block_tokens=4,groups=groups)
    assert torch.equal(actual.counts,expected.counts)
    torch.testing.assert_close(actual.key,expected.key,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(actual.value,expected.value,atol=2e-6,rtol=2e-6)
    assert actual.counts.sum() == 2*3*(14-len(selected))


def test_reject_unknown_duplicate_and_out_of_range_coordinates():
    k = torch.ones(1,7,1,8)
    frame = build_frame_moments(k,k,block_tokens=4)
    for f,t in [([8],[0]),([5],[7]),([5,5],[2,2])]:
        with pytest.raises(ValueError):
            gather_moment_payload([frame],[5],torch.tensor([[f]]),torch.tensor([[t]]))


def test_frame_moments_do_not_alias_original_kv():
    k = torch.ones(1,7,1,8)
    frame = build_frame_moments(k,k,block_tokens=4)
    before = frame.key_sum.clone()
    k.zero_()
    assert torch.equal(before,frame.key_sum)
    assert frame.bytes > 0
