from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from adapters.longlive_sparse.native_semantic_remat import NativeSemanticRematMemory


def pipe():
    cache=dict(k=torch.zeros(1,32,1,1),v=torch.zeros(1,32,1,1),global_end_index=torch.tensor(96),
        local_end_index=torch.tensor(32),pinned_start=torch.tensor(8),pinned_len=torch.tensor(8))
    return SimpleNamespace(use_relative_rope=False,guidance_scale=1,quantize_kv=False,num_frame_per_block=8,
        sampling_steps=4,local_attn_size=32,sink_size=8,global_sink_size=8,frame_seq_length=1,
        kv_cache_pos=[cache],kv_cache_neg=[],crossattn_cache_pos=[{'is_init':True}],crossattn_cache_neg=[],
        _dit_model=SimpleNamespace(local_attn_size=32,t_scale=1,rope_method='linear',original_seq_len=128,
                                  use_relative_rope=False,rope_temporal_offset=24))


@pytest.mark.parametrize('condition,expected',[('past',2.),('current',7.)])
def test_isolated_reencode_reuses_storage_restores_clock_and_uses_declared_condition(monkeypatch,condition,expected):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.random,'fork_rng',lambda **kwargs:nullcontext())
    p=pipe();c=p.kv_cache_pos[0];ptrs=c['k'].data_ptr(),c['v'].data_ptr()
    m=NativeSemanticRematMemory(p,reconstruction_condition=condition,mode='raw_reveal',source_end=48,target_start=96,
        prompts=[],destination='global',anchor_policy='source_repeat_pinned')
    m.source_latent=torch.ones(1,8,1);m.source_condition=torch.full((1,2,1),2.)
    m.current_condition=torch.full((1,2,1),7.)
    m.source_settings=dict(local_attn_size=32,t_scale=1,rope_method='linear',original_seq_len=128,
                           use_relative_rope=False,rope_temporal_offset=8)
    m.capture=dict(source_frames=list(range(40,48)),latent_sha256='source')
    calls=[]
    def generator(**kwargs):
        assert m.busy and int(c['local_end_index'])==0 and int(c['global_end_index'])==40
        assert p._dit_model.rope_temporal_offset==8 and not p.crossattn_cache_pos[0]['is_init']
        assert kwargs['current_start']==40 and (kwargs['timestep']==0).all()
        calls.append(float(kwargs['conditional_dict']['prompt_embeds'][0,0,0]))
        c['k'][:,:8].fill_(calls[-1]);c['v'][:,:8].fill_(calls[-1]+1)
        c['global_end_index'].fill_(48);c['local_end_index'].fill_(8)
    p.generator=generator;m._install()
    assert calls==[expected] and ptrs==(c['k'].data_ptr(),c['v'].data_ptr())
    assert (c['k'][:,:16]==expected).all() and (c['v'][:,:16]==expected+1).all()
    assert int(c['global_end_index'])==96 and int(c['local_end_index'])==16
    assert int(c['pinned_start'])==8 and p._dit_model.rope_temporal_offset==24
    assert not p.crossattn_cache_pos[0]['is_init'] and not m.busy
    assert m.install['not_exact_original_KV_replay']


def test_reentrant_callbacks_do_not_relabel_outer_query():
    p=pipe();m=NativeSemanticRematMemory(p,reconstruction_condition='past',mode='raw_reveal',source_end=48,target_start=96,
        prompts=[],destination='global',anchor_policy='source_repeat_pinned')
    m.active_start=96;m.busy=True
    m.before(None,(),{'current_start':40})
    assert m.active_start==96 and not m.counts
