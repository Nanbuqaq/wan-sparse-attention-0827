from dataclasses import replace

import pytest
import torch

from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.phase_prototypes import canonical_wan_frequency_table, archive_rope0_key
from adapters.longlive_sparse.selectors import summarize_query_for_pretransfer
from adapters.longlive_sparse.memory_dynamics import compare_coordinates


def test_aligned_index_preserves_raw_kv_and_matches_isolated_reference():
    cfg=SparseHistoryConfig(method='rope_aligned_final_history',history_density=.25,
                           refresh_policy='per_chunk')
    archive=HistoryArchive(cfg,spatial_height=8,spatial_width=16)
    control_cfg=SparseHistoryConfig(method='transfer_vaware_hybrid_history',history_density=.25,
        method_params={'base_fraction':.7,'local_fraction':.15,'transfer_multiplier':1.,'v_weight':1.,'query_block_size':64})
    reference=HistoryArchive(control_cfg,spatial_height=8,spatial_width=16)
    frequency=canonical_wan_frequency_table(64)
    for frame in (1,2):
        k,v=torch.randn(1,128,2,64),torch.randn(1,128,2,64)
        stored=archive.index_frame(0,frame,k,v)
        roped=archive_rope0_key(k,spatial_height=8,spatial_width=16,freqs=frequency)
        expected=reference.index_frame(0,frame,roped,v,storage_k=k,storage_v=v)
        assert torch.equal(stored.key,k) and torch.equal(stored.value,v)
        assert torch.equal(stored.block_centroids,expected.block_centroids)
        assert torch.equal(stored.block_value_centroids,expected.block_value_centroids)
    q=summarize_query_for_pretransfer(torch.randn(1,128,2,64),64,coordinate_space='post_rope')
    selected=archive.route_indexed(0,q,[1,2],exact_k_tokens=128)
    control=reference.route_indexed(0,q,[1,2],exact_k_tokens=128)
    assert compare_coordinates(selected,control,token_base=128)['jaccard']==1
    assert selected.digest()!=control.digest()  # distinct method identity
    assert selected.metadata['key_prototype_space']=='spatial_rope0'
    assert archive.online_routing_context(0,q,[1,2]).metadata['query_summary_space']=='post_rope'
    assert len(archive._prototype_freqs)==1
    with pytest.raises(ValueError,match='post_rope'):
        archive.route_indexed(0,replace(q,coordinate_space='unrotated'),[1,2],exact_k_tokens=128)
    archive.clear_frames()
    assert not archive._prototype_freqs


def test_aligned_method_rejects_unvalidated_position_policy():
    with pytest.raises(ValueError,match='upstream_zero'):
        SparseHistoryConfig(method='rope_aligned_final_history',rope_policy='recency_rank')


def test_matched_suite_freezes_dense_and_sparse_budgets_without_capture():
    from scripts.build_aligned_final_probe import build
    from scripts.run_loaded_method_suite import _history_density
    suite,expected=build('a'*40)
    assert len(expected['cases'])==6
    assert not suite['formal_prompts_used']
    for case in suite['cases']:
        assert not case['complete_capture']
        assert _history_density(suite,case,'rag_dense')==1.
        assert _history_density(suite,case,'rope_aligned_final_history')==.25
    assert _history_density(suite,{'history_density':.5},'rag_dense')==.5
