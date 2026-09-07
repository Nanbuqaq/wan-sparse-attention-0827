import pytest
import torch

from adapters.longlive_sparse.contexts import OnlineRoutingContext
from adapters.longlive_sparse.group_relations import summarize_groups, build_group_relation_route


@pytest.mark.parametrize('grouping', ['random_balanced', 'spatial_quadrants', 'query_features'])
def test_query_groups_cover_exactly_and_are_causal(grouping):
    q = torch.randn(1, 32, 2, 8, generator=torch.Generator().manual_seed(71))
    a = summarize_groups(q, grouping=grouping, spatial_height=4, spatial_width=4)
    b = summarize_groups(q, grouping=grouping, spatial_height=4, spatial_width=4)
    assert torch.equal(a.query_labels, b.query_labels)
    assert torch.equal(a.query_group_sizes.sum(-1), torch.full((1, 2), 32))
    for h in range(2):
        for group in range(4):
            mask = a.query_labels[0, h] == group
            if mask.any():
                torch.testing.assert_close(a.query_centroids[0, h, group], q[0, mask, h].mean(0))


def test_conditional_groups_can_expand_union_without_changing_pairs():
    labels = torch.tensor([[[0, 0, 1, 1, 2, 2, 3, 3]]])
    context = OnlineRoutingContext(query_centroids=torch.eye(4).view(1, 1, 4, 4)*10,
        query_group_sizes=torch.full((1, 1, 4), 2), key_prototypes=torch.eye(4).view(1, 1, 4, 4)*10,
        value_prototypes=torch.ones(1, 1, 4, 4), block_frame_ids=torch.zeros(4, dtype=torch.long),
        block_token_starts=torch.arange(4)*4, block_token_ends=(torch.arange(4)+1)*4,
        block_age=torch.zeros(4))
    frames, tokens = torch.zeros(1, 1, 16, dtype=torch.long), torch.arange(16).view(1, 1, -1)
    routes = [build_group_relation_route(context, labels, frames, tokens, exact_tokens=8,
        grouping='test', admission=admission, density=.25) for admission in ('shared', 'per_group')]
    assert routes[0].history_pairs == routes[1].history_pairs == 32
    assert routes[0].unique_history_tokens == 4
    assert routes[1].unique_history_tokens == 16
    assert routes[0].digest() != routes[1].digest()


@pytest.mark.parametrize('layer', [0, 9])
def test_online_runtime_route_cannot_read_full_candidate_values(layer):
    from dataclasses import replace
    from adapters.longlive_sparse.archive import HistoryArchive
    from adapters.longlive_sparse.config import SparseHistoryConfig
    from adapters.longlive_sparse.selectors import summarize_query_for_pretransfer
    config = SparseHistoryConfig(method='group_relation_history', refresh_policy='per_chunk',
                                 history_density=.25, method_params={'group_start_layer': 8})
    archive = HistoryArchive(config, spatial_height=4, spatial_width=4)
    q = torch.randn(1, 32, 2, 8)
    class ShapeOnly:
        shape = (1, 16, 2, 8)
        dtype = torch.float32  # Storage metadata is online-legal; tensor values are not.
        def element_size(self):
            return 4
        def __getattr__(self, name):
            raise AssertionError('online route attempted full KV access: ' + name)
    for frame in range(1, 5):
        key, value = torch.randn(1, 16, 2, 8), torch.randn(1, 16, 2, 8)
        item = archive.index_frame(layer, frame, key, value)
        archive._layers[layer][frame] = replace(item, key=ShapeOnly(), value=ShapeOnly())
    summary = (summarize_query_for_pretransfer(q, 64) if layer < 8 else
               summarize_groups(q, grouping='spatial_quadrants', spatial_height=4, spatial_width=4))
    route = archive.route_group_relation(layer, summary, [1, 2, 3, 4], exact_k_tokens=16)
    assert route.method == 'group_relation_history'
    assert route.history_pair_density == .25
    assert route.metadata['routing_identity']['active'] == (layer >= 8)
