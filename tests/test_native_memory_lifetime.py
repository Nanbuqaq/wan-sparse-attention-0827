from types import SimpleNamespace

import pytest
import torch

from adapters.longlive_sparse.native_episode_memory import NativeEpisodeMemory, admission_plan
from scripts.run_native_memory_study import ARMS, arm_arguments


def fake_pipe():
    cache = dict(k=torch.arange(32).reshape(1,32,1,1).float(),
                 v=torch.arange(32).reshape(1,32,1,1).float()+100,
                 global_end_index=torch.tensor(96), local_end_index=torch.tensor(32),
                 pinned_start=torch.tensor(8), pinned_len=torch.tensor(8))
    return SimpleNamespace(use_relative_rope=False, guidance_scale=1, quantize_kv=False,
        num_frame_per_block=8, sampling_steps=4, local_attn_size=32, sink_size=8,
        global_sink_size=8, frame_seq_length=1, kv_cache_pos=[cache], kv_cache_neg=[],
        crossattn_cache_pos=[], crossattn_cache_neg=[],
        _dit_model=SimpleNamespace(local_attn_size=32,t_scale=1,rope_method='default',
            original_seq_len=128,use_relative_rope=False,rope_temporal_offset=24))


def test_lifetime_changes_admission_not_storage_semantics():
    spec=dict(source_end=48,target_start=96,frames=8,frame_tokens=880,layers=30,
              heads=24,dim=128,dtype=torch.bfloat16)
    old, old_sha=admission_plan(**spec)
    new, new_sha=admission_plan(**spec,restore_after_frames=8)
    assert new.pop('restore_displaced_global_at_latent') == 104
    assert new == old and new_sha != old_sha


@pytest.mark.parametrize('mode,destination,duration', [
    ('none','global',8),('log_reveal','global',8),('raw_reveal','shot',8),('raw_reveal','global',7)])
def test_rejects_undefined_lifetime_interventions(mode,destination,duration):
    with pytest.raises(ValueError):
        NativeEpisodeMemory(fake_pipe(),mode=mode,source_end=48,target_start=96,
            prompts=[],destination=destination,restore_after_frames=duration)


def test_one_chunk_restore_is_exact_and_fully_charged(monkeypatch):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    pipe=fake_pipe();cache=pipe.kv_cache_pos[0]
    original_k=cache['k'].clone();original_v=cache['v'].clone()
    memory=NativeEpisodeMemory(pipe,mode='raw_reveal',source_end=48,target_start=96,
        prompts=[],restore_after_frames=8)
    memory.bank=[(torch.full((1,8,1,1),77.),torch.full((1,8,1,1),88.))]
    memory.capture={'source_frames':list(range(40,48))}
    memory.ledger['CPU_archive_peak_bytes']=64
    memory.before(None,(),{'current_start':96})
    assert (cache['k'][:,:8]==77).all() and torch.equal(cache['k'][:,8:],original_k[:,8:])
    assert memory.ledger['CPU_archive_peak_bytes']==128
    # A second denoising call does not reinstall or prematurely expire memory.
    memory.before(None,(),{'current_start':96})
    assert memory.ledger['demand_H2D_payload_bytes']==64
    # The newly generated return context is not rolled back with the old prefix.
    cache['k'][:,24:]=123
    memory.before(None,(),{'current_start':104})
    assert torch.equal(cache['k'][:,:8],original_k[:,:8])
    assert torch.equal(cache['v'],original_v) and (cache['k'][:,24:]==123).all()
    assert memory.ledger['rollback_D2H_payload_bytes']==64
    assert memory.ledger['restore_H2D_payload_bytes']==64
    assert memory.audit()['restoration']['at_latent']==104
    memory.before(None,(),{'current_start':104})
    assert memory.ledger['restore_H2D_payload_bytes']==64


def test_large_window_does_not_attach_local32_episode_object():
    args=arm_arguments('window128',gate=True)
    assert '--episode-memory-mode' not in args and '--native-local-frames' in args
    assert '--episode-gate-layout' in args
    assert len(ARMS)==5
