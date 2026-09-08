from types import SimpleNamespace

import pytest
import torch

from adapters.longlive_sparse.native_attention_teacher import NativeAttentionTeacherCapture,spatial_query_indices


def capture(budget=10**6):
    return NativeAttentionTeacherCapture(SimpleNamespace(frame_seq_length=16,sampling_steps=4),
        query_frame=96,token_grid=(4,4),budget=budget)


def test_sites_are_per_frame_and_include_state_lower_center():
    indices=spatial_query_indices(22,40,8)
    assert len(indices)==32 and len(set(indices))==32
    assert indices[:4]==[460,210,230,660]
    assert all(f*880<=i<(f+1)*880 for f in range(8) for i in indices[f*4:f*4+4])


def test_capture_is_independent_copy_and_preserves_original_call():
    observer=capture();observer.before(None,(),{'current_start':96*16})
    q=torch.arange(32,dtype=torch.bfloat16).reshape(1,32,1,1);k=q.clone();v=q.clone();marker=object()
    def original(a,b,c):
        assert a is q and b is k and c is v
        return marker
    assert observer.observe(original,q,k,v) is marker
    assert len(observer.records)==1 and observer.records[0]['layer']==0
    k.zero_()
    assert observer.records[0]['k'].count_nonzero()==31
    assert observer.records[0]['q'].shape[1]==8


def test_capture_budget_is_checked_before_copy():
    observer=capture(1);observer.before(None,(),{'current_start':96*16})
    q=torch.ones(1,16,1,1,dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):observer.observe(lambda q,k,v:q,q,q,q)
    assert observer.bytes==0 and not observer.records


def test_incomplete_grid_cannot_be_exported(tmp_path):
    with pytest.raises(RuntimeError):capture().export(tmp_path/'incomplete.pt')


def test_unselected_calls_preserve_all_options():
    observer=capture();observer.before(None,(),{'current_start':0})
    marker=object()
    def original(q,k,v,*,option):
        assert option==7
        return marker
    assert observer.observe(original,None,None,None,option=7) is marker
