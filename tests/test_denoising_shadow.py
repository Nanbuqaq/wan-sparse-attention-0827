from types import SimpleNamespace

import pytest
import torch

from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.phase_prototypes import canonical_wan_frequency_table
from adapters.longlive_sparse.selectors import summarize_query_for_pretransfer
from scripts.evaluate_denoising_shadow import evaluate
from scripts.summarize_memory_dynamics import distribution


def test_cached_call_capture_marks_fresh_shadow_and_replay_checks_five_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    from adapters.longlive_sparse.runtime_attention import SparseHistorySelfAttention
    monkeypatch.setenv('LONGLIVE_CAPTURE_COMPLETE_ATTENTION', '1')
    monkeypatch.setenv('LONGLIVE_COMPLETE_CAPTURE_LAYERS', '0')
    monkeypatch.setenv('LONGLIVE_COMPLETE_CAPTURE_STARTS', '640')
    monkeypatch.setenv('LONGLIVE_COMPLETE_CAPTURE_PASSES', '5')
    cfg=SparseHistoryConfig(method='transfer_vaware_hybrid_history', refresh_policy='per_chunk',
        method_params={'base_fraction':.7,'local_fraction':.15,'query_block_size':64,'v_weight':1.,'transfer_multiplier':1.})
    archive=HistoryArchive(cfg,spatial_height=8,spatial_width=16)
    for frame in (1,2):
        archive.index_frame(0,frame,torch.randn(1,128,1,64),torch.randn(1,128,1,64))
    q=torch.randn(1,64,1,64)
    summary=summarize_query_for_pretransfer(q,64)
    plan=archive.route_indexed(0,summary,[1,2],exact_k_tokens=64)
    module=SimpleNamespace(layer_id=0,_complete_capture_counts={},history_archive=archive,sparse_config=cfg,
                           _capture_root=lambda kind:tmp_path)
    for index in range(5):
        current=q+index*.3
        SparseHistorySelfAttention._capture_complete_attention(module,current_start=640,
            query=current,query_unrotated=current,exact_key=q,exact_value=q,
            global_frame_ids=torch.tensor([1,2]),freqs=canonical_wan_frequency_table(64),
            frame_seqlen=128,route_plan=plan,route_summary=summary if index==0 else None)
    paths=sorted(tmp_path.glob('*.pt'))
    result=evaluate(paths,device='cpu')
    assert result['status']=='pass' and len(result['records'])==5
    assert result['records'][0]['records']['fresh_per_call']['vs_executed_route']['max_abs']==0
    assert result['records'][1]['query_summary_source']=='recomputed_after_route_for_diagnostic_only'
    assert result['records'][-1]['phase']=='clean_context_commit'
    from scripts.audit_candidate_permutation import audit
    order_audit=audit(torch.load(paths[0],weights_only=True))
    assert all(row['same_candidate_set'] for row in order_audit['rows'])
    assert order_audit['rows'][0]['same_route_sha']
    assert order_audit['rows'][0]['same_logical_edges']
    from scripts.evaluate_proxy_refresh_factorial import construct_factorial
    factorial=construct_factorial(torch.load(paths[0],weights_only=True),
                                  torch.load(paths[-1],weights_only=True),device='cpu')
    assert set(factorial)=={'raw_first','raw_current','aligned_first','aligned_current'}
    assert len({plan.unique_history_tokens for plan in factorial.values()})==1
    with pytest.raises(ValueError,match='five-call|four denoising'):
        evaluate(paths[:1],device='cpu')


def test_distribution_uses_interpolated_even_median_and_explicit_empty():
    assert distribution([0.,1.])['median']==.5
    assert distribution([])['median'] is None


def test_render_never_appends_unaffected_baseline_future_after_fork():
    from scripts.render_memory_pulses import build_fork_latents
    baseline=torch.zeros(1,120,2,2,2)
    suffix=torch.ones(1,9,2,2,2)
    combined=build_fork_latents(baseline,suffix)
    assert combined.shape[1]==39
    assert torch.equal(combined[:,:30],baseline[:,:30])
    assert torch.equal(combined[:,30:],suffix)
