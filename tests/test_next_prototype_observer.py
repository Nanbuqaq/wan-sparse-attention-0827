from types import SimpleNamespace
import torch
from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.prefetch_context_probe import NextPrototypeObserver
from adapters.longlive_sparse.selectors import summarize_query_for_pretransfer


def test_next_layer_snapshot_uses_only_already_indexed_frames_and_owns_values():
    config = SparseHistoryConfig(method='transfer_vaware_hybrid_history', history_density=.25)
    archive = HistoryArchive(config, spatial_height=2, spatial_width=32)
    for layer, frames in ((0, (1,2)), (1, (1,))):
        for frame in frames:
            k,v = torch.randn(1,64,2,16),torch.randn(1,64,2,16)
            archive.index_frame(layer, frame, k, v)
    q = torch.randn(1,64,2,16)
    summary = summarize_query_for_pretransfer(q,64)
    route = archive.route_indexed(0,summary,[1,2],exact_k_tokens=64)
    probe = NextPrototypeObserver(starts=(46800,), layers=(0,1))
    probe(module=SimpleNamespace(layer_id=0, history_archive=archive), query=q, route_plan=route,
        candidate_frame_ids=torch.tensor([1,2]), current_start=46800, denoising_pass=0, route_was_reused=False)
    record = probe.records[0]
    assert record['target_frames_already_indexed'] == [1]
    assert record['target_missing_at_prediction'] == [2]
    saved = record['next_context']['key_prototypes'].clone()
    archive._layers[1][1].block_centroids.fill_(999.)
    assert torch.equal(saved,record['next_context']['key_prototypes'])
    assert not record['target_Q_or_route_used']
    assert 'query' not in record['next_context'] and 'key' not in record['next_context']


def test_q_to_next_predictor_respects_causal_available_blocks_and_physical_cap():
    from adapters.longlive_sparse.contexts import OnlineRoutingContext
    from scripts.analyze_remaining_contracts import q_to_next_prediction
    from scripts.probe_verified_prefetch_routes import width
    starts=torch.arange(25)*64
    context=OnlineRoutingContext(query_centroids=torch.ones(1,1,2,4),query_group_sizes=torch.tensor([[[64,8]]]),
        key_prototypes=torch.ones(1,1,25,4),value_prototypes=torch.ones(1,1,25,4),
        block_frame_ids=torch.ones(25,dtype=torch.long),block_token_starts=starts,
        block_token_ends=(starts+64).clamp_max(1560),block_age=torch.zeros(25))
    predicted=q_to_next_prediction(context,[1])
    assert predicted and {p[1] for p in predicted}=={1}
    assert sum(width(p) for p in predicted)<=int(1560*.25)
