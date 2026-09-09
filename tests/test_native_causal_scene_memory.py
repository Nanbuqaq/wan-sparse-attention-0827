from types import SimpleNamespace

import torch

from adapters.longlive_sparse.native_causal_scene_memory import NativeCausalSceneMemory


def fake_pipe():
    g=torch.Generator().manual_seed(9)
    cache=dict(k=torch.randn(1,32,1,128,generator=g).bfloat16(),v=torch.randn(1,32,1,128,generator=g).bfloat16(),
        global_end_index=torch.tensor(48),local_end_index=torch.tensor(32),pinned_start=torch.tensor(8),pinned_len=torch.tensor(8))
    return SimpleNamespace(use_relative_rope=False,guidance_scale=1,quantize_kv=False,num_frame_per_block=8,
        sampling_steps=4,local_attn_size=32,sink_size=8,global_sink_size=8,frame_seq_length=1,
        kv_cache_pos=[cache],kv_cache_neg=[],crossattn_cache_pos=[],crossattn_cache_neg=[],
        _dit_model=SimpleNamespace(local_attn_size=32,t_scale=1,rope_method='linear',original_seq_len=None,
            use_relative_rope=False,rope_temporal_offset=8))


def test_closed_scene_banks_are_owned_and_FIFO_budget_is_explicit(monkeypatch):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    pipe=fake_pipe();memory=NativeCausalSceneMemory(pipe,archive_budget=2*(4096+8))
    for frame in (48,56,64):
        pipe.kv_cache_pos[0]['global_end_index'].fill_(frame)
        memory.last_commit=dict(end=frame,phase=8,prototype=torch.tensor([1.,0.]))
        memory._archive_last_scene(frame)
    assert [b['descriptor'].source_end for b in memory.banks]==[56,64]
    assert memory.ledger['evicted_archives']==1 and memory.ledger['CPU_archive_peak_tensor_bytes']==8208
    saved=memory.banks[-1]['kv'][0][0].clone()
    pipe.kv_cache_pos[0]['k'].zero_()
    assert torch.equal(saved,memory.banks[-1]['kv'][0][0])


def test_dynamic_source_choice_installs_once_and_preserves_bank_values(monkeypatch):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    pipe=fake_pipe();memory=NativeCausalSceneMemory(pipe)
    memory.last_commit=dict(end=48,phase=8,prototype=torch.tensor([1.,0.]))
    memory._archive_last_scene(48)
    key,value=(t.clone() for t in memory.banks[0]['kv'][0])
    pipe._dit_model.rope_temporal_offset=24
    pipe.kv_cache_pos[0]['global_end_index'].fill_(96)
    memory.last_commit=dict(end=96,phase=16,prototype=torch.tensor([0.,1.]))
    condition={'prompt_embeds':torch.tensor([[[1.,0.]]])}
    memory.before(None,(),{'current_start':96,'conditional_dict':condition},current_text='Back to the jar.')
    assert len(memory.installations)==1
    plan=memory.installations[0]['installation']['admission_plan']
    assert plan['source_frames']==list(range(40,48)) and plan['temporal_delta']==64
    assert plan['archive_version']==1
    assert torch.equal(pipe.kv_cache_pos[0]['v'][:,8:16],value)
    assert torch.equal(memory.banks[0]['kv'][0][0],key)
    assert memory.ledger['history_H2D_KV_bytes']==4096
    assert int(pipe.kv_cache_pos[0]['global_end_index'])==96
    memory.last_commit=dict(end=104,phase=24,prototype=torch.tensor([1.,0.]))
    memory.before(None,(),{'current_start':104,'conditional_dict':condition},current_text='Back to the jar.')
    assert len(memory.installations)==1 and memory.decisions[-1]['selected_version'] is None
