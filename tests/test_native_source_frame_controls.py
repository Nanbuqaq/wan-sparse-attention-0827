from types import SimpleNamespace
import pytest
import torch
from adapters.longlive_sparse.native_causal_block_memory import CausalBlockConfig,NativeCausalBlockMemory,source_frame_indices,gather_source_heads
from adapters.longlive_sparse.native_causal_block_memory import repeat_source_frames


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


def test_reconstruction_repeats_whole_frames_and_owns_an_independent_buffer():
    source=torch.arange(2*3*2*4).reshape(1,6,2,4).bfloat16()
    actual=repeat_source_frames(source,3,4)
    expected=torch.cat([source[:,:3]]*4+[source[:,3:]]*4,dim=1)
    assert torch.equal(actual,expected) and actual.untyped_storage().data_ptr()!=source.untyped_storage().data_ptr()
    CausalBlockConfig(policy='frame_uniform',fraction=.25,source_repeats=4)
    with pytest.raises(ValueError):CausalBlockConfig(policy='frame_recent',fraction=.25,source_repeats=4)
    with pytest.raises(ValueError):CausalBlockConfig(policy='mass_value',fraction=.25,source_repeats=4)


def test_reconstruction_preserves_protected_slots_and_reuses_installed_source():
    from collections import defaultdict
    runtime=object.__new__(NativeCausalBlockMemory)
    runtime.frame_tokens=2;runtime.target_frame=48;runtime.active_start=96;runtime.active_phase=0
    runtime.config=CausalBlockConfig(policy='frame_uniform',fraction=.25,source_repeats=4)
    runtime.masks={};runtime.rows=[];runtime.routes_saved=[];runtime.calls=1
    runtime.ledger=defaultdict(float);runtime.binding={'temporal_delta':0.}
    raw=torch.arange(16*2*128).reshape(1,16,2,128).bfloat16()
    runtime.active_bank={'kv':[(raw,raw+1)],'descriptor':SimpleNamespace(archive_version=2,source_end=16)}
    cache={k:torch.full((1,64,2,128),-1000.,dtype=torch.bfloat16) for k in ('k','v')}
    runtime.pipe=SimpleNamespace(kv_cache_pos=[cache]);q=torch.zeros(1,16,2,128,dtype=torch.bfloat16)
    expected=torch.cat([raw[:,4:6]]*4+[raw[:,12:14]]*4,1)
    def original(q,k,v):
        assert k.shape[1]==64 and torch.equal(k[:,16:32],expected)
        assert torch.all(k[:,:16]==-1000) and torch.all(k[:,32:]==-1000)
        return q
    kwargs=dict(info={'pinned_shift':0},current_start=96,window_start=0,effective_sink=32,
        pinned_start=16,pinned_len=16,prepend_sink=False,prepend_pinned=False,max_tokens=64,cache_end=64,global_sink_tokens=16)
    runtime.dispatch(0,original,q,cache['k'],cache['v'],**kwargs)
    transferred=runtime.ledger['source_KV_H2D_bytes'];writes=runtime.ledger['source_repeat_cache_write_logical_bytes']
    runtime.active_phase=1;runtime.dispatch(0,original,q,cache['k'],cache['v'],**kwargs)
    assert runtime.ledger['source_KV_H2D_bytes']==transferred and runtime.ledger['source_repeat_cache_write_logical_bytes']==writes
    assert runtime.rows[0]['selected_source_tokens_per_head']==16
    assert runtime.rows[0]['raw_unique_source_tokens_per_head']==4
