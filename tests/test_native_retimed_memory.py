from types import SimpleNamespace

import torch
import pytest

from adapters.longlive_sparse.native_retimed_memory import NativeRetimedEpisodeMemory,binding_coordinates


@pytest.mark.parametrize('policy,delta,virtual_start,phase',[
    ('recent_virtual',64,88,24),('phase_only',16,40,24),('age_only',48,88,8)])
def test_retimming_changes_only_temporal_channels_and_explicit_version(monkeypatch,policy,delta,virtual_start,phase):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    c=dict(k=torch.zeros(1,32,1,128),v=torch.zeros(1,32,1,128),global_end_index=torch.tensor(96),
           local_end_index=torch.tensor(32),pinned_start=torch.tensor(8),pinned_len=torch.tensor(8))
    p=SimpleNamespace(use_relative_rope=False,guidance_scale=1,quantize_kv=False,num_frame_per_block=8,
        sampling_steps=4,local_attn_size=32,sink_size=8,global_sink_size=8,frame_seq_length=1,
        kv_cache_pos=[c],kv_cache_neg=[],crossattn_cache_pos=[],crossattn_cache_neg=[],
        _dit_model=SimpleNamespace(local_attn_size=32,t_scale=1,rope_method='linear',original_seq_len=None,
                                  use_relative_rope=False,rope_temporal_offset=24))
    m=NativeRetimedEpisodeMemory(p,mode='raw_reveal',source_end=48,target_start=96,prompts=[],destination='shot',position_policy=policy)
    k=torch.randn(1,8,1,128);v=torch.randn_like(k);m.bank=[(k.clone(),v.clone())]
    m.capture={'source_temporal_offset':8,'source_frames':list(range(40,48))}
    m.before(None,(),{'current_start':96})
    assert torch.equal(c['v'][:,8:16],v) and torch.equal(c['k'][:,8:16,...,44:],k[...,44:])
    assert not torch.equal(c['k'][:,8:16,...,:44],k[...,:44])
    assert int(c['global_end_index'])==96 and int(c['local_end_index'])==32
    plan=m.install['admission_plan']
    assert plan['temporal_delta']==delta and plan['virtual_source_frames']==list(range(virtual_start,virtual_start+8))
    assert plan['bound_phase']==phase and plan['age_delta']+plan['phase_delta']==delta
    assert p._dit_model.rope_temporal_offset==24
    assert m.audit()['temporal_rebinding_is_algorithm_not_layout']


@pytest.mark.parametrize('policy,expected',[('recent_virtual',40),('phase_only',8),('age_only',32)])
def test_wrong_source_has_its_own_age_and_phase(policy,expected):
    coordinates=binding_coordinates(source_frame=56,target_frame=96,frames=8,source_phase=16,current_phase=24,policy=policy)
    assert coordinates['temporal_delta']==expected


def test_invalid_policy_and_future_coordinates_rejected():
    with pytest.raises(ValueError):
        binding_coordinates(source_frame=40,target_frame=96,frames=8,source_phase=8,current_phase=24,policy='unknown')
    with pytest.raises(ValueError):
        binding_coordinates(source_frame=96,target_frame=96,frames=8,source_phase=8,current_phase=24,policy='recent_virtual')
