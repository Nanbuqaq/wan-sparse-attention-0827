from types import SimpleNamespace
import torch
import pytest
from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.phase_prototypes import canonical_wan_frequency_table
from adapters.longlive_sparse.selectors import summarize_query_for_pretransfer
from scripts.prepare_cross_trajectory_routes import construct,METHODS
from scripts.replay_cross_trajectory import evaluate


@pytest.mark.parametrize('actor',[*METHODS,'rag_dense'])
def test_cross_routes_reproduce_both_actor_spaces_and_keep_budget(tmp_path,monkeypatch,actor):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    from adapters.longlive_sparse.runtime_attention import SparseHistorySelfAttention
    monkeypatch.setenv('LONGLIVE_CAPTURE_COMPLETE_ATTENTION','1')
    monkeypatch.setenv('LONGLIVE_COMPLETE_CAPTURE_LAYERS','0')
    monkeypatch.setenv('LONGLIVE_COMPLETE_CAPTURE_STARTS','640')
    monkeypatch.setenv('LONGLIVE_COMPLETE_CAPTURE_PASSES','5')
    params={'base_fraction':.7,'local_fraction':.15,'v_weight':1.,'transfer_multiplier':1.,'query_block_size':64}
    cfg=SparseHistoryConfig(method=actor,history_density=1. if actor=='rag_dense' else .25,
                           refresh_policy='per_chunk',method_params={} if actor=='rag_dense' else params)
    archive=HistoryArchive(cfg,spatial_height=8,spatial_width=16)
    for frame in [1,2]:archive.index_frame(0,frame,torch.randn(1,128,2,64),torch.randn(1,128,2,64))
    raw=torch.randn(1,128,2,64);post=raw+.2
    space='post_rope' if actor==METHODS[1] else 'unrotated'
    summary=summarize_query_for_pretransfer(post if space=='post_rope' else raw,64,coordinate_space=space)
    plan=(archive.full_history_route(0,[1,2],query_shape=raw.shape,exact_k_tokens=128) if actor=='rag_dense'
          else archive.route_indexed(0,summary,[1,2],exact_k_tokens=128))
    module=SimpleNamespace(layer_id=0,_complete_capture_counts={},history_archive=archive,sparse_config=cfg,
                           _capture_root=lambda kind:tmp_path)
    for call in range(5):
        SparseHistorySelfAttention._capture_complete_attention(module,current_start=640,query=post+call*.1,
            query_unrotated=raw+call*.1,exact_key=post,exact_value=raw,global_frame_ids=torch.tensor([1,2]),
            freqs=canonical_wan_frequency_table(64),frame_seqlen=128,route_plan=plan,
            route_summary=summary if call==0 and actor!='rag_dense' else None)
    captures=[torch.load(p,weights_only=True) for p in sorted(tmp_path.glob('*.pt'))]
    plans,checks=construct(captures[0],device='cpu')
    assert set(plans)==set(METHODS)
    assert plans[METHODS[0]].unique_history_tokens==plans[METHODS[1]].unique_history_tokens
    if actor in METHODS:assert plans[actor].digest()==plan.digest() and len(checks)==2
    result=evaluate(captures,plans,actor=actor,device='cpu')
    assert result['status']=='pass' and len(result['rows'])==5
    captures[-1]['key']=captures[-1]['key']+1
    with pytest.raises(ValueError,match='history changed'):evaluate(captures,plans,actor=actor,device='cpu')
