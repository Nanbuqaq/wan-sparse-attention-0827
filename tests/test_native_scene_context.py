from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from adapters.longlive_sparse.native_episode_memory import NativeEpisodeMemory
from adapters.longlive_sparse.native_scene_context import NativeSceneContextReset


def pipe():
    cache=dict(k=torch.arange(32).reshape(1,32,1,1).float(),v=torch.ones(1,32,1,1),
        global_end_index=torch.tensor(96),local_end_index=torch.tensor(32),
        pinned_start=torch.tensor(8),pinned_len=torch.tensor(8))
    return SimpleNamespace(use_relative_rope=False,guidance_scale=1,quantize_kv=False,
        num_frame_per_block=8,sampling_steps=4,local_attn_size=32,sink_size=8,
        global_sink_size=8,frame_seq_length=1,kv_cache_pos=[cache],kv_cache_neg=[],
        crossattn_cache_pos=[],crossattn_cache_neg=[],
        _dit_model=SimpleNamespace(local_attn_size=32,t_scale=1,rope_method='linear',
            original_seq_len=128,use_relative_rope=False,rope_temporal_offset=24))


def test_reset_excludes_suffix_without_zero_filling_or_corrupting_clock(monkeypatch):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    native=pipe();other=deepcopy(native)
    args=dict(mode='raw_reveal',source_end=48,target_start=96,prompts=[],destination='shot')
    ordinary=NativeEpisodeMemory(other,**args);reset=NativeSceneContextReset(native,**args)
    for memory in (ordinary,reset):
        memory.bank=[(torch.full((1,8,1,1),77.),torch.full((1,8,1,1),88.))]
        memory.capture={'source_frames':list(range(40,48))};memory.ledger['CPU_archive_peak_bytes']=64
        memory.before(None,(),{'current_start':96})
    c=native.kv_cache_pos[0]
    assert torch.equal(c['k'][:,:8],torch.arange(8).reshape(1,8,1,1).float())
    assert (c['k'][:,8:16]==77).all()
    assert torch.equal(c['k'][:,16:],torch.arange(16,32).reshape(1,16,1,1).float())
    assert int(c['local_end_index'])==16 and int(c['global_end_index'])==96
    assert int(c['pinned_start'])==8 and int(c['pinned_len'])==8
    assert reset.ledger['demand_H2D_payload_bytes']==ordinary.ledger['demand_H2D_payload_bytes']==64
    assert reset.install['admission_plan_sha256']!=ordinary.install['admission_plan_sha256']
    assert reset.audit()['logical_context_change_not_layout_optimization']


@pytest.mark.parametrize('mode,destination,ttl',[('none','shot',0),('log_reveal','shot',0),('raw_reveal','global',0),('raw_reveal','shot',8)])
def test_rejects_unregistered_reset(mode,destination,ttl):
    with pytest.raises(ValueError):NativeSceneContextReset(pipe(),mode=mode,source_end=48,
        target_start=96,prompts=[],destination=destination,restore_after_frames=ttl)


def test_shape_observer_forwards_inputs_and_output_unchanged():
    memory=NativeSceneContextReset(pipe(),mode='raw_reveal',source_end=48,
        target_start=96,prompts=[],destination='shot')
    memory.active_start=96
    q=torch.ones(1,8,1,1);k=torch.ones(1,24,1,1);v=torch.zeros_like(k)
    marker=object()
    def original(a,b,c,*,flag):
        assert a is q and b is k and c is v and flag
        return marker
    assert memory.observe_attention(original,q,k,v,flag=True) is marker
    assert memory.attention_shapes[(96,8,24)]==1


@pytest.mark.parametrize('policy,active,multiplicity',[('source_only',8,1),('source_repeat',16,2)])
def test_source_anchor_has_explicit_multiplicity_clock_and_D2D_ledger(monkeypatch,policy,active,multiplicity):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    p=pipe();m=NativeSceneContextReset(p,mode='raw_reveal',source_end=48,target_start=96,
        prompts=[],destination='global',anchor_policy=policy)
    m.bank=[(torch.full((1,8,1,1),77.),torch.full((1,8,1,1),88.))]
    m.capture={'source_frames':list(range(40,48))};m.ledger['CPU_archive_peak_bytes']=64
    m.before(None,(),{'current_start':96});c=p.kv_cache_pos[0]
    assert (c['k'][:,:active]==77).all() and int(c['global_end_index'])==96
    assert int(c['local_end_index'])==active
    assert int(c['pinned_start'])==(-1 if policy=='source_only' else 8)
    assert m.install['admission_plan']['source_multiplicity']==multiplicity
    assert m.install['admission_plan']['visible_history_source_frames']==list(range(40,48))*multiplicity
    assert m.ledger['demand_H2D_payload_bytes']==64
    assert m.ledger['source_replication_D2D_bytes']==(0 if policy=='source_only' else 64)
