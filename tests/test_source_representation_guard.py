from types import SimpleNamespace
import pytest
import torch
from adapters.longlive_sparse.native_shared_conditioning import SharedNativeConditioning


def fixture():
    encoder=SimpleNamespace(values={'past':torch.ones(1,3,5),'now':torch.zeros(1,3,5)},aliases={},target_device='cpu')
    shared=SharedNativeConditioning(encoder);shared.encode(encoder,[['past','past','now']],1)
    def cache(start):
        return dict(k=torch.zeros(1,8,1,2),v=torch.zeros(1,8,1,2),local_end_index=torch.tensor(0),global_end_index=torch.tensor(start))
    calls=[]
    def forward(**kw):calls.append(kw);return 'ok'
    pipe=SimpleNamespace(frame_seq_length=1,kv_cache_pos=[cache(16)],crossattn_cache_pos=[dict(k=torch.zeros(1,3,1,2),v=torch.zeros(1,3,1,2))],generator=SimpleNamespace(forward=forward))
    return shared,pipe,[cache(8)],[dict(k=torch.zeros(1,3,1,2),v=torch.zeros(1,3,1,2))],calls


def test_causal_source_and_current_conditions_without_future_or_copy_acceptance():
    shared,pipe,caches,cross,calls=fixture();x=torch.zeros(1,8,48,2,2,dtype=torch.bfloat16)
    for kind,index in [('past',1),('current',2)]:
        condition=shared.auxiliary_source_condition(source_start=8,current_start=16,kind=kind)
        assert condition['prompt_embeds'].data_ptr()==shared.blocks[index]['prompt_embeds'].data_ptr()
        assert shared.run_source_auxiliary(pipe,x,condition,caches,cross,source_start=8,current_start=16,kind=kind)=='ok'
    assert len(calls)==2 and shared.ledger['generator_calls_verified']==0
    assert shared.ledger['auxiliary_source_calls_verified']==2
    for source,current in [(16,16),(24,16),(1,16),(8,24)]:
        with pytest.raises(ValueError):shared.auxiliary_source_condition(source_start=source,current_start=current,kind='past')
    bad=dict(prompt_embeds=shared.blocks[2]['prompt_embeds'].clone())
    with pytest.raises(RuntimeError):shared.run_source_auxiliary(pipe,x,bad,caches,cross,source_start=8,current_start=16,kind='current')


def test_auxiliary_cannot_alias_live_native_storage_or_change_shape():
    shared,pipe,caches,cross,calls=fixture();x=torch.zeros(1,8,48,2,2,dtype=torch.bfloat16)
    condition=shared.auxiliary_source_condition(source_start=8,current_start=16,kind='current')
    caches[0]['k']=pipe.kv_cache_pos[0]['k']
    with pytest.raises(RuntimeError,match='aliases'):shared.run_source_auxiliary(pipe,x,condition,caches,cross,source_start=8,current_start=16,kind='current')
    assert calls==[]


def test_ordinary_condition_guard_remains_strict_after_auxiliary_call():
    shared,pipe,caches,cross,calls=fixture();x=torch.zeros(1,8,48,2,2,dtype=torch.bfloat16)
    shared.run_source_auxiliary(pipe,x,shared.blocks[1],caches,cross,source_start=8,current_start=16,kind='past')
    with pytest.raises(RuntimeError):shared.verify_consumption(1,None,(),dict(current_start=16,conditional_dict=shared.blocks[1]))
    shared.verify_consumption(1,None,(),dict(current_start=16,conditional_dict=shared.blocks[2]))
