from types import SimpleNamespace
import pytest
import torch
from adapters.longlive_sparse.native_causal_block_memory import CausalBlockConfig,NativeCausalBlockMemory,source_frame_indices,gather_source_heads


@pytest.mark.parametrize('policy,frames',[('frame_recent',[6,7]),('frame_uniform',[2,6])])
def test_quarter_keeps_two_complete_source_frames_for_every_head(policy,frames):
    ids=source_frame_indices(880,1760,policy)
    assert ids.unique().numel()==1760 and (ids.reshape(2,880)//880)[:,0].tolist()==frames
    raw=torch.arange(7040*2*4).reshape(1,7040,2,4)
    actual=gather_source_heads(raw,ids[None].repeat(2,1))
    expected=torch.cat([raw[:,f*880:(f+1)*880] for f in frames],1)
    assert torch.equal(actual,expected)
    assert torch.equal(source_frame_indices(880,7040,policy),torch.arange(7040))


def test_frame_controls_do_not_build_group_summaries():
    for policy in ('frame_recent','frame_uniform'):
        runtime=object.__new__(NativeCausalBlockMemory);calls=[]
        runtime.config=CausalBlockConfig(policy=policy,fraction=.25)
        runtime.scene=SimpleNamespace(_archive_last_scene=lambda frame:calls.append(frame))
        runtime._sample_memory=lambda stage:None
        runtime._archive_with_groups(48)
        assert calls==[48]
    with pytest.raises(ValueError):CausalBlockConfig(policy='frame_recent',fraction=.2)
    with pytest.raises(ValueError):source_frame_indices(880,1761,'frame_recent')
