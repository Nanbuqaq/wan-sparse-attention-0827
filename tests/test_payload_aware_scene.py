import torch
from types import SimpleNamespace
import math
from adapters.longlive_sparse.payload_aware_scene import choose_available,PayloadAwareScene
from adapters.longlive_sparse.native_scene_admission import SceneDescriptor,choose_scene


def test_evicted_best_descriptor_abstains_instead_of_falling_to_weaker_live_bank():
    a=SceneDescriptor(1,8,0.,torch.tensor([1.,0.]));b=SceneDescriptor(2,16,8.,torch.tensor([.9,(1-.9**2)**.5]))
    query=torch.tensor([1.,0.]);text='Back to the original object.'
    assert choose_scene(text,query,[b],64)['selected_version']==2
    result=choose_available(text,query,[a,b],{2},64)
    assert result['selected_version'] is None and result['metadata_selected_version']==1
    assert result['reason']=='selected_payload_evicted_abstain'
    assert choose_available(text,query,[a,b],{1,2},64)['selected_version']==1


def test_raw_eviction_keeps_only_small_descriptor_and_respects_owned_bytes(monkeypatch):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    cache=dict(k=torch.zeros(1,32,2,4,dtype=torch.bfloat16),v=torch.zeros(1,32,2,4,dtype=torch.bfloat16),local_end_index=8,global_end_index=8)
    pipe=SimpleNamespace(use_relative_rope=False,guidance_scale=1,quantize_kv=False,num_frame_per_block=8,local_attn_size=32,sink_size=8,frame_seq_length=1,kv_cache_pos=[cache])
    memory=PayloadAwareScene(pipe,archive_budget=528)
    for i,end in enumerate((8,16,24)):
        cache['global_end_index']=end;cache['k'].fill_(i);cache['v'].fill_(i+1)
        memory.last_commit=dict(end=end,phase=float(i*8),prototype=torch.tensor([1.,0.]))
        memory._archive_last_scene(end)
    assert [x['descriptor'].archive_version for x in memory.banks]==[2,3]
    assert [x.archive_version for x in memory.catalog]==[1,2,3]
    assert sum(x['owned_bytes'] for x in memory.banks)==528
    assert memory.audit()['catalog'][0]['payload_available'] is False
    assert memory.audit()['catalog_prototype_bytes']==24


def test_descriptor_lineage_and_ranking_are_independent_controls():
    def desc(version,end,score):return SceneDescriptor(version,end,0.,torch.tensor([score,math.sqrt(1-score*score)]))
    a=desc(2,32,.84);b=desc(3,64,.863);derived=desc(4,88,.876)
    canonical=SceneDescriptor(4,88,24.,a.condition_prototype);query=torch.tensor([1.,0.])
    text='Back to the same ceramic jug.'
    assert choose_available(text,query,[a,b,derived],{2,3,4},120)['selected_version']==4
    assert choose_available(text,query,[a,b,canonical],{2,3,4},120)['selected_version']==4
    assert choose_available(text,query,[a,b,derived],{2,3,4},120,ranking='max_similarity')['selected_version']==4
    assert choose_available(text,query,[a,b,canonical],{2,3,4},120,ranking='max_similarity')['selected_version']==3
    assert choose_available(text,query,[a,b,canonical],{4},120,ranking='max_similarity')['reason']=='selected_payload_evicted_abstain'
