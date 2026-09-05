import torch
from adapters.longlive_sparse.ar_routing import build_route_plan


def test_lower_pair_density_can_increase_grouped_executor_storage():
    common = dict(method='fixture', routing_stage='pre-transfer',
                  history_frame_ids=torch.ones(1,1,4,dtype=torch.long),
                  history_token_ids=torch.arange(4).view(1,1,4),
                  candidate_history_tokens=4, exact_k_tokens=8, density=1., metadata={})
    shared = build_route_plan(**common, query_labels=torch.tensor([[[0,0,1,1]]]),
                             selections=[[[torch.tensor([0,1,2,3]), torch.tensor([0,1,2,3])]]])
    sparse = build_route_plan(**common, query_labels=torch.tensor([[[0,0,1,1]]]),
                             selections=[[[torch.tensor([0,1]),torch.tensor([2,3])]]])
    assert sparse.history_pairs < shared.history_pairs
    assert sparse.unique_history_tokens == shared.unique_history_tokens
    dense_storage = shared.grouped_executor_storage(head_dim=64, element_size=2)
    sparse_storage = sparse.grouped_executor_storage(head_dim=64, element_size=2)
    assert sparse_storage['one_packed_kv_bytes'] > dense_storage['one_packed_kv_bytes']
    assert sparse_storage['measured_hbm_transactions'] is False
