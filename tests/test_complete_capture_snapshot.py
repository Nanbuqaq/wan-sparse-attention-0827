from types import SimpleNamespace
import torch
from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.phase_prototypes import canonical_wan_frequency_table
from adapters.longlive_sparse.selectors import summarize_query_for_pretransfer


def test_actual_online_snapshot_is_weights_only_serializable(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    from adapters.longlive_sparse.runtime_attention import SparseHistorySelfAttention
    monkeypatch.setenv('LONGLIVE_CAPTURE_COMPLETE_ATTENTION','1')
    monkeypatch.setenv('LONGLIVE_COMPLETE_CAPTURE_LAYERS','0')
    monkeypatch.setenv('LONGLIVE_COMPLETE_CAPTURE_STARTS','640')
    cfg=SparseHistoryConfig(method='transfer_vaware_hybrid_history',refresh_policy='per_chunk')
    archive=HistoryArchive(cfg,spatial_height=8,spatial_width=16)
    for frame in (1,2):
        archive.index_frame(0,frame,torch.randn(1,128,2,64),torch.randn(1,128,2,64))
    q=torch.randn(1,128,2,64)
    summary=summarize_query_for_pretransfer(q,64)
    plan=archive.route_indexed(0,summary,[1,2],exact_k_tokens=128)
    module=SimpleNamespace(layer_id=0,_complete_capture_counts={},history_archive=archive,sparse_config=cfg,
                           _capture_root=lambda kind:tmp_path)
    SparseHistorySelfAttention._capture_complete_attention(module,current_start=640,query=q,query_unrotated=q,
        exact_key=q,exact_value=q,global_frame_ids=torch.tensor([1,2]),freqs=canonical_wan_frequency_table(64),
        frame_seqlen=128,route_plan=plan,route_summary=summary)
    captured=torch.load(tmp_path/'layer00_start00000640_pass00.pt',weights_only=True)
    assert captured['schema_version']==3
    assert torch.equal(captured['actual_query_summary']['query_centroids'],summary.query_centroids)
    assert captured['actual_online_context']['value_prototypes'].shape==(1,2,4,64)
