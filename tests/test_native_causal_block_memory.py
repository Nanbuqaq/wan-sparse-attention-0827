import torch
from adapters.longlive_sparse.native_causal_block_memory import groups_for_source,gather_source_heads,exact_source_indices,source_head_scores
from adapters.longlive_sparse.native_temporal_rephase import rephase_temporal_keys
from adapters.longlive_sparse.native_causal_block_memory import NativeCausalBlockMemory,CausalBlockConfig
from collections import defaultdict
from types import SimpleNamespace


def test_partitions_cover_real880_and_gate128_without_padding_tokens():
    for h,w in ((22,40),(8,16)):
        for kind in ('flat64','spatial8','flat_matched'):
            groups=groups_for_source(h,w,kind=kind)
            assert sorted(t for g in groups for t in g)==list(range(8*h*w))
            assert all(0<len(g)<=64 for g in groups)


def test_flat_matched_controls_spatial_group_count_size_and_temporal_span():
    spatial=groups_for_source(22,40,kind='spatial8')
    flat=groups_for_source(22,40,kind='flat_matched')
    assert [len(g) for g in spatial]==[len(g) for g in flat]
    assert spatial!=flat
    for left,right in zip(spatial,flat):
        assert {i//880 for i in left}=={i//880 for i in right}


def test_memory_report_survives_native_cache_release_before_final_video_audit(monkeypatch):
    runtime=object.__new__(NativeCausalBlockMemory)
    runtime.pipe=SimpleNamespace(kv_cache_pos=None)
    runtime.device=torch.device('cuda:0')
    runtime.scene=SimpleNamespace(banks=[{'owned_bytes':123}])
    runtime.memory_samples=[]
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda device:456)
    monkeypatch.setattr(torch.cuda,'memory_reserved',lambda device:789)
    runtime._sample_memory('after_video')
    assert runtime.memory_samples[0]['GPU_allocated_bytes']==456
    assert runtime.memory_samples[0]['CPU_archive_tensor_bytes']==123


def test_constructor_uses_generator_device_before_lazy_KV_allocation():
    model=torch.nn.Linear(1,1)
    model.blocks=[SimpleNamespace(self_attn=SimpleNamespace(_research_inplace_cache=True))]
    model.t_scale=1;model.rope_method='linear';model.original_seq_len=None
    pipe=SimpleNamespace(use_relative_rope=False,guidance_scale=1,quantize_kv=False,
        num_frame_per_block=8,sampling_steps=4,generator=SimpleNamespace(_compiled_model_call=None),
        frame_seq_length=128,_dit_model=model,local_attn_size=32,global_sink_size=8,sink_size=8,kv_cache_pos=None)
    runtime=NativeCausalBlockMemory(pipe,CausalBlockConfig(),(8,16))
    assert runtime.device==model.weight.device and pipe.kv_cache_pos is None


def test_head_specific_gather_preserves_each_original_coordinate_and_dtype():
    raw=torch.arange(1*17*3*4).reshape(1,17,3,4).bfloat16()
    ids=torch.tensor([[0,3,9,16],[1,4,5,10],[2,7,8,12]])
    actual=gather_source_heads(raw,ids)
    for head in range(3):
        for j,index in enumerate(ids[head]):assert torch.equal(actual[0,j,head],raw[0,index,head])
    assert actual.is_contiguous() and actual.dtype==raw.dtype


def test_selected_key_rebinding_matches_gathering_rebound_original_keys():
    rng=torch.Generator().manual_seed(17)
    k=torch.randn(1,19,3,128,generator=rng).bfloat16()
    ids=torch.tensor([[0,3,8,18],[2,5,9,16],[1,4,7,12]])
    a=rephase_temporal_keys(gather_source_heads(k,ids),51.5)
    b=gather_source_heads(rephase_temporal_keys(k,51.5),ids)
    assert torch.equal(a,b)


def test_per_head_scores_do_not_pool_heads_and_fixed_source_budget_is_exact():
    q=torch.zeros(4,2,128);km=torch.zeros(3,2,128);vm=torch.zeros_like(km)
    vm[0,0]=1;vm[2,1]=2
    scores=source_head_scores(q,km,vm,torch.ones(3),'mass_value')
    assert scores.argmax(-1).tolist()==[0,2]
    groups=groups_for_source(22,40,kind='spatial8')
    ids=exact_source_indices(groups,[0.]*len(groups),1760)
    assert len(ids)==len(set(ids))==1760


def test_partial_source_dispatch_never_attends_stale_unused_slots():
    runtime=object.__new__(NativeCausalBlockMemory)
    runtime.frame_tokens=2;runtime.target_frame=48;runtime.active_start=96
    runtime.config=CausalBlockConfig(policy='random',fraction=.5,head_policy='per_head')
    runtime.groups=groups_for_source(1,2);runtime.masks={};runtime.rows=[];runtime.routes_saved=[]
    runtime.ledger=defaultdict(float);runtime.calls=1;runtime.binding={'temporal_delta':0.}
    raw=torch.arange(16*2*128).reshape(1,16,2,128).bfloat16()
    runtime.active_bank={'kv':[(raw,raw+1)],'descriptor':SimpleNamespace(archive_version=2,source_end=16)}
    cache={'k':torch.full((1,64,2,128),-1000.,dtype=torch.bfloat16),
           'v':torch.full((1,64,2,128),-1000.,dtype=torch.bfloat16)}
    runtime.pipe=SimpleNamespace(kv_cache_pos=[cache])
    q=torch.zeros(1,16,2,128,dtype=torch.bfloat16);seen={}
    def original(q,k,v):seen.update(k=k.clone(),v=v.clone());return q
    result=runtime.dispatch(0,original,q,cache['k'],cache['v'],info={'pinned_shift':0},
        current_start=96,window_start=0,effective_sink=32,pinned_start=16,pinned_len=16,
        prepend_sink=False,prepend_pinned=False,max_tokens=64,cache_end=64,global_sink_tokens=16)
    assert result is q and seen['k'].shape[1]==56
    ids=runtime.routes_saved[0]['source_indices'].long()
    assert torch.equal(seen['k'][:,16:24],gather_source_heads(raw,ids))
    assert torch.equal(seen['v'][:,16:24],gather_source_heads(raw+1,ids))
    # Remaining stale source slots still exist physically but are not executed.
    assert torch.all(cache['k'][:,24:32]==-1000)
    assert torch.all(seen['k'][:,:16]==-1000) and torch.all(seen['k'][:,24:]==-1000)
    assert runtime.rows[0]['selected_source_tokens_per_head']==8
