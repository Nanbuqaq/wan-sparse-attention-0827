from types import SimpleNamespace
import pytest
import torch
from adapters.longlive_sparse.native_scene_release import NativeSceneRelease,scene_positions


def controller():
    cache=torch.arange(128,dtype=torch.bfloat16).reshape(1,16,2,4)
    pipe=SimpleNamespace(use_relative_rope=False,guidance_scale=1,quantize_kv=False,num_frame_per_block=8,sampling_steps=4,
        generator=SimpleNamespace(_compiled_model_call=None),frame_seq_length=1,local_attn_size=16,
        _dit_model=SimpleNamespace(blocks=[None],rope_temporal_offset=0.),kv_cache_pos=[dict(k=cache,v=cache+1)])
    result=NativeSceneRelease(pipe,'w2_scene_release',current_text=lambda frame:'A white flower.' if frame==16 else 'Back to the same bowl.')
    result.owners[0]=[('native',i,5,0.) for i in range(16)];result.active_start=16
    return result


def test_only_current_phase_and_unknown_binding_rejected():
    owners=[('native',i,5,0. if i<4 else 8.) for i in range(8)]
    assert scene_positions(owners,list(range(8)),8.,True)==[4,5,6,7]
    assert scene_positions(owners,list(range(8)),8.,False)==list(range(8))
    owners[0]=None
    with pytest.raises(ValueError):scene_positions(owners,list(range(8)),8.,True)


def test_retired_copy_is_independent_deduplicated_and_bounded():
    c=controller();old=c.pipe.kv_cache_pos[0]['k'].clone();size=c.archive_resident()
    c.pipe.kv_cache_pos[0]['k'].zero_()
    assert torch.equal(c.retired[(0,c.owners[0][5])][0],old[:,5:6])
    assert c.archive_resident()==0 and c.retired_bytes==size
    audit=c.audit()
    assert audit['selector']=='current_scene_owner_filter'
    assert not audit['no_new_archive_without_recall'] and audit['recalled_chunk'] is None
    c=controller();c.retired_budget=1
    with pytest.raises(RuntimeError,match='16GiB'):c.archive_resident()
    assert not c.retired
    c=controller();c.active_start=8
    with pytest.raises(RuntimeError,match='future'):c.archive_resident()
    assert not c.retired


def test_nonreturn_cut_retires_and_current_return_cue_releases_filter():
    c=controller();c.last_phase=0.;c.pipe._dit_model.rope_temporal_offset=8.
    c.before(None,None,dict(current_start=16,timestep=torch.tensor([1])))
    assert c.release_active and c.retired_bytes>0 and c.release_events[-1]['frame']==16
    stored=c.retired_bytes;c.pipe._dit_model.rope_temporal_offset=16.
    c.before(None,None,dict(current_start=24,timestep=torch.tensor([1])))
    assert not c.release_active and c.retired_bytes==stored
    assert c.release_events[-1]['current_text_return_cue'] and not c.release_events[-1]['archive_restored']
