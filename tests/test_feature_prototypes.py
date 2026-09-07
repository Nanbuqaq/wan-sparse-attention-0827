import pytest
import torch
from adapters.longlive_sparse.feature_prototypes import build_feature_tail
from adapters.longlive_sparse.prototype_tail import execute_weighted_tail_sdpa
from adapters.longlive_sparse.offline_eval import dense_history_attention


@pytest.mark.parametrize('grouping',['spatial_groups','random_groups','key_kmeans'])
def test_group_counts_and_constant_key_exactness(grouping):
    gen=torch.Generator().manual_seed(531)
    q=torch.randn(1,5,2,8,generator=gen)
    k=torch.randn(1,14,2,8,generator=gen)
    k[:,:4]=k[:,:1]; k[:,4:7]=k[:,4:5]; k[:,7:11]=k[:,7:8]; k[:,11:]=k[:,11:12]
    v=torch.randn(k.shape,generator=gen)
    indices=torch.empty(1,2,0,dtype=torch.long)
    tail,info=build_feature_tail(k,v,indices,frame_tokens=7,block_tokens=4,groups=2,grouping=grouping)
    assert torch.equal(tail.counts.sum(-1),torch.full((1,2),14.))
    empty=k[:,:0]
    output=execute_weighted_tail_sdpa(q,empty,empty,empty,empty,tail)
    expected=dense_history_attention(q,k,v)
    torch.testing.assert_close(output,expected,atol=2e-6,rtol=2e-6)
    assert info['prototype_slots']==8
