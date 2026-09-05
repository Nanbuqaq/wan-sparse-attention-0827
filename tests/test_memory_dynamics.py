from types import SimpleNamespace
import torch
import pytest

from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.memory_dynamics import compare_coordinates, frame_lifecycle, MemoryDynamicsObserver
from adapters.longlive_sparse.checkpoint_probe import freeze_tree, thaw_tree, choose_pulse_indices
from adapters.longlive_sparse.selectors import summarize_query_for_pretransfer
from adapters.longlive_sparse.system_config import LongLiveSystemConfig
from adapters.longlive_sparse.route_plan import HistoryRoutePlan


def _plan(token):
    return HistoryRoutePlan(method='block64_history', routing_stage='pre-transfer',
        query_labels=torch.tensor([[[0,0]]]), query_group_sizes=torch.tensor([[[2]]]),
        union_frame_ids=torch.tensor([[[1]]]), union_token_ids=torch.tensor([[[token]]]),
        group_union_indices=torch.tensor([[[[0]]]]), group_history_counts=torch.tensor([[[1]]]),
        candidate_history_tokens=4, query_tokens=2, exact_k_tokens=2, target_history_density=.25)


def test_coordinates_keep_owner_and_partial_block_semantics():
    a, b = _plan(0), _plan(1)
    assert compare_coordinates(a, b, token_base=4)['jaccard'] == 0
    assert compare_coordinates(a, b, token_base=4, block_size=64)['jaccard'] == 1
    b.union_frame_ids += 1
    assert compare_coordinates(a, b, token_base=4, block_size=64)['jaccard'] == 0
    with pytest.raises(ValueError, match='geometry'):
        compare_coordinates(a, _plan(9), token_base=4)


def test_lifecycle_is_coarse_causal_and_censored():
    rows = [{'query_frame': 18, 'num_evicted': 3, 'selected_global_frames': [[1, 2]]},
            {'query_frame': 21, 'num_evicted': 4, 'selected_global_frames': [[2, 3]]},
            {'query_frame': 24, 'num_evicted': 5, 'selected_global_frames': [[1, 3]]}]
    result = frame_lifecycle(rows, sink_size=1, recent_exclude=1, chunk_frames=3)
    assert result['adjacent_frame_jaccard'] == [1/3, 1/3]
    assert sorted(result['revisit_gap_chunks']) == [1., 1., 2.]
    assert result['never_selected_within_observation'] == 1
    assert result['right_censored'] and not result['semantic_importance_claim']
    with pytest.raises(ValueError, match='duplicate chunk'):
        frame_lifecycle(rows + rows[:1], sink_size=1, recent_exclude=1, chunk_frames=3)


def test_shadow_observation_never_mutates_executed_route(tmp_path):
    cfg = SparseHistoryConfig(method='transfer_vaware_hybrid_history', refresh_policy='per_chunk',
        method_params={'base_fraction':.7, 'local_fraction':.15, 'v_weight':1., 'transfer_multiplier':1.})
    archive = HistoryArchive(cfg, spatial_height=8, spatial_width=16)
    for frame in (1, 2):
        archive.index_frame(0, frame, torch.randn(1,128,2,64), torch.randn(1,128,2,64))
    q = torch.randn(1,128,2,64)
    plan = archive.route_indexed(0, summarize_query_for_pretransfer(q,64), [1,2], exact_k_tokens=128)
    digest = plan.digest()
    module = SimpleNamespace(layer_id=0, sparse_config=cfg, system_config=LongLiveSystemConfig(), history_archive=archive)
    observer = MemoryDynamicsObserver(tmp_path, layers=[0], shadow_starts=[640])
    observer.current_timestep = 500
    observer(module=module, query=q, route_plan=plan, candidate_frame_ids=torch.tensor([1,2]),
             current_start=640, denoising_pass=0, route_was_reused=False)
    observer.current_timestep = 0
    observer(module=module, query=q+2, route_plan=plan, candidate_frame_ids=torch.tensor([1,2]),
             current_start=640, denoising_pass=4, route_was_reused=True)
    assert plan.digest() == digest
    assert observer.records[-1]['fresh_shadow_computed']
    assert observer.records[-1]['phase'] == 'clean_context_commit'
    assert len(observer.finish()) == 64


def test_checkpoint_tree_isolation_and_equal_cardinality_pulses():
    original = {'k': torch.arange(3), 'metadata': [1, ('x',)]}
    frozen = freeze_tree(original)
    a, b = thaw_tree(frozen), thaw_tree(frozen)
    a['k'][0] = 99
    assert b['k'][0] == original['k'][0] == 0
    reference = torch.tensor([[2, 0]])
    assert choose_pulse_indices(reference, eligible_count=5, policy='newest', seed=3).tolist() == [[3,4]]
    assert choose_pulse_indices(reference, eligible_count=5, policy='oldest', seed=3).tolist() == [[0,1]]
    assert torch.equal(choose_pulse_indices(reference, eligible_count=5, policy='random', seed=3),
                       choose_pulse_indices(reference, eligible_count=5, policy='random', seed=3))
    assert torch.equal(choose_pulse_indices(reference, eligible_count=5, policy='reference', seed=3), reference)
