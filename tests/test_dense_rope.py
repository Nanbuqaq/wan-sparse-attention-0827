import pytest
import torch
from adapters.longlive_sparse.dense_rope import direct_output_causal_rope
from adapters.longlive_sparse.rope import apply_selected_rope
from adapters.longlive_sparse.phase_prototypes import canonical_wan_frequency_table
from adapters.longlive_sparse.system_config import LongLiveSystemConfig


@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float32,torch.float64])
def test_direct_rope_matches_sparse_coordinate_reference_and_keeps_padding(dtype):
    torch.manual_seed(131)
    x=torch.randn(2,24,3,64,dtype=dtype)
    grids=torch.tensor([[2,3,4],[1,2,5]])
    freqs=canonical_wan_frequency_table(64)
    actual=direct_output_causal_rope(x,grids,freqs,start_frame=3)
    for b,(f,h,w) in enumerate(grids.tolist()):
        t=torch.arange(f*h*w)
        pos=torch.stack((t//(h*w)+3,t//w%h,t%w),-1)[None,None].expand(1,3,-1,-1)
        expected=apply_selected_rope(x[b:b+1,:f*h*w],pos,freqs)
        assert torch.equal(actual[b:b+1,:f*h*w],expected)
        assert torch.equal(actual[b,f*h*w:],x[b,f*h*w:])


def test_explicit_relative_indices_and_noncontiguous_input():
    x=torch.randn(1,24,2,128)[...,::2]
    freqs=canonical_wan_frequency_table(64)
    out=direct_output_causal_rope(x,torch.tensor([[2,3,4]]),freqs,relative_frame_indices=torch.tensor([5,1]))
    assert torch.equal(out[:,:12],direct_output_causal_rope(x[:,:12],torch.tensor([[1,3,4]]),freqs,start_frame=5))
    assert out.is_contiguous()


def test_rope_layout_enters_system_identity_and_invalid_values_rejected():
    assert LongLiveSystemConfig().identity_dict()!=LongLiveSystemConfig(local_rope_layout='direct_output').identity_dict()
    with pytest.raises(ValueError): LongLiveSystemConfig(local_rope_layout='fp32_fast')
