import pytest
import torch

from adapters.longlive_sparse.ar_routing import build_route_plan
from adapters.longlive_sparse.attention_bias import AttentionBiasPlan
from adapters.longlive_sparse.backends import execute_plan
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.oracle_tether import build_oracle_bias


def full_route(query_tokens=7, history_tokens=6, heads=2):
    return build_route_plan(method='oracle_test', routing_stage='post-transfer',
        query_labels=torch.zeros(1, heads, query_tokens, dtype=torch.long),
        selections=[[[torch.arange(history_tokens)] for _ in range(heads)]],
        history_frame_ids=torch.ones(1, heads, history_tokens, dtype=torch.long),
        history_token_ids=torch.arange(history_tokens).view(1, 1, -1).expand(1, heads, -1),
        candidate_history_tokens=history_tokens, exact_k_tokens=3, density=1., metadata={})


def test_split_query_reference_matches_dense_bias_without_qk_mask_storage():
    torch.manual_seed(18)
    q = torch.randn(1, 7, 2, 8)
    ek, ev = torch.randn(1, 3, 2, 8), torch.randn(1, 3, 2, 8)
    k, v = torch.randn(1, 6, 2, 8), torch.randn(1, 6, 2, 8)
    route = full_route()
    qr = torch.nn.functional.one_hot(torch.tensor([[0, 1, 1, 0, 1, 0, 1]]), 2).float()
    kr = torch.nn.functional.one_hot(torch.tensor([[[0, 1, 0, 1, 1, 0], [1, 1, 0, 0, 1, 0]]]), 2).float()
    bias = AttentionBiasPlan(('identity', 'scene'), qr, kr, torch.rand(1, 2, 6)+.05,
                              mode='oracle_test', metadata={'context_weight': .25})
    compact = execute_plan('split_role_sdpa_reference', q, ek, ev, k, v, route, bias)
    dense = execute_plan('biased_sdpa_reference', q, ek, ev, k, v, route, bias)
    torch.testing.assert_close(compact.output, dense.output, atol=1e-6, rtol=1e-5)
    assert compact.logical_pairs == dense.logical_pairs


def test_oracle_time_alignment_is_explicit_and_cannot_use_sparse_transfer():
    pixel = torch.zeros(153, 30, 52)
    for frame in range(153):
        pixel[frame, :, :frame % 52] = 1
    masks = {'pixel_patch_masks': pixel, 'latent_anchor_masks': pixel[::4]}
    route = full_route(query_tokens=1560, history_tokens=1560, heads=2)
    source = build_oracle_bias(route, masks, timeline='source_compatible_addressing',
        current_latent=18, sink_frames=1, cpu_pool_frames=20)
    aligned = build_oracle_bias(route, masks, timeline='aligned_latent_anchors',
        current_latent=18, sink_frames=1, cpu_pool_frames=20)
    assert source.digest() != aligned.digest()
    assert not torch.equal(source.query_role_probabilities, aligned.query_role_probabilities)
    assert not torch.equal(source.history_role_probabilities, aligned.history_role_probabilities)
    assert source.metadata['online_method'] is False
    with pytest.raises(ValueError, match='full KV'):
        SparseHistoryConfig(method='tethermem_oracle_mask_teacher', history_density=.25,
                            backend='split_role_sdpa_reference')
