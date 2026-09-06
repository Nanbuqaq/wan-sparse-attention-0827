from dataclasses import replace
import torch
import pytest
from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.selectors import summarize_query_for_pretransfer
from adapters.longlive_sparse.memory_dynamics import compare_coordinates


@pytest.mark.parametrize('bootstrap_layer',[-1,9])
@pytest.mark.parametrize('layer',[0,9])
def test_bootstrap_matches_exact_raw_or_aligned_reference_and_preserves_archive(bootstrap_layer,layer):
    params={'base_fraction':.7,'local_fraction':.15,'v_weight':1.,'transfer_multiplier':1.,'query_block_size':64}
    cfg=SparseHistoryConfig(method='rope_bootstrap_ablation_history',method_params={**params,'bootstrap_layer':bootstrap_layer})
    archive=HistoryArchive(cfg,spatial_height=8,spatial_width=16)
    raw=HistoryArchive(SparseHistoryConfig(method='transfer_vaware_hybrid_history',method_params=params),spatial_height=8,spatial_width=16)
    aligned=HistoryArchive(SparseHistoryConfig(method='rope_aligned_final_history',method_params=params),spatial_height=8,spatial_width=16)
    for frame in [1,2]:
        k,v=torch.randn(1,128,2,64),torch.randn(1,128,2,64)
        a=archive.index_frame(layer,frame,k,v)
        r=raw.index_frame(layer,frame,k,v);p=aligned.index_frame(layer,frame,k,v)
        assert torch.equal(a.key,k) and torch.equal(a.value,v)
        assert torch.equal(a.block_centroids,p.block_centroids)
        assert (a.block_unrotated_centroids is not None)==cfg.needs_bootstrap_raw(layer)
        if cfg.needs_bootstrap_raw(layer):
            assert torch.equal(a.block_unrotated_centroids,r.block_centroids)
            assert a.index_bytes-p.index_bytes==a.block_unrotated_centroids.numel()*4
    q=torch.randn(1,128,2,64)
    for ids in [[1],[1,2]]:
        phase=cfg.uses_aligned_prototypes(layer,len(ids))
        space='post_rope' if phase else 'unrotated'
        summary=summarize_query_for_pretransfer(q,64,coordinate_space=space)
        plan=archive.route_indexed(layer,summary,ids,exact_k_tokens=128)
        reference=(aligned if phase else raw).route_indexed(layer,summary,ids,exact_k_tokens=128)
        assert compare_coordinates(plan,reference,token_base=128)['jaccard']==1.
        assert plan.metadata['routing_identity']['active_space']==space
        ctx=archive.online_routing_context(layer,summary,ids)
        assert ctx.metadata['key_prototype_space']==('spatial_rope0' if phase else 'unrotated')
        with pytest.raises(ValueError,match='prototype policy'):
            archive.route_indexed(layer,replace(summary,coordinate_space='unrotated' if phase else 'post_rope'),ids,exact_k_tokens=128)


def test_ablation_suite_freezes_four_distinct_interventions():
    from scripts.build_bootstrap_ablation import build
    suite,expected=build('b'*40)
    assert len(expected['cases'])==4 and len({c['id'] for c in expected['cases']})==4
    assert [c['lane'] for c in expected['cases']]==[0,1,0,1]
    assert {c['method_params']['bootstrap_layer'] for c in suite['cases']}=={-1,9}
    assert suite['capture_starts']==[28080] and suite['capture_passes']==1
