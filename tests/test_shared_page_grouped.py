import torch
from adapters.longlive_sparse.ar_routing import build_route_plan
from adapters.longlive_sparse.grouped_staging import build_grouped_packing_recipe
from adapters.longlive_sparse.shared_page_grouped import build_shared_page_recipe


def test_shared_pages_reconstruct_exact_per_group_sequence_and_dedupe_prefix():
    labels = (torch.arange(36) % 3).view(1, 1, -1)
    frames = torch.zeros(1, 1, 128, dtype=torch.long)
    tokens = torch.arange(128).view(1, 1, -1)
    plan = build_route_plan(method='test', routing_stage='pre-transfer', query_labels=labels,
        selections=[[[torch.arange(g*16, g*16+32) for g in range(3)]]],
        history_frame_ids=frames, history_token_ids=tokens, candidate_history_tokens=128,
        exact_k_tokens=600, density=.25, metadata={})
    union = plan.union_frame_ids.shape[-1]
    old = build_grouped_packing_recipe(plan, exact_tokens=600, union_tokens=union)
    new = build_shared_page_recipe(plan, exact_tokens=600, union_tokens=union)
    pages = new.packing.key_indices.view(-1, 256)
    cursor = 0
    for row, length in enumerate(old.key_lengths):
        reconstructed = pages[new.block_table[row].long()].reshape(-1)[:length]
        assert torch.equal(reconstructed, old.key_indices[cursor:cursor+length])
        cursor += length
    assert torch.equal(new.block_table[0, :2], new.block_table[1, :2])
    assert new.physical_tokens < new.logical_packed_tokens
