import torch
import pytest
from adapters.longlive_sparse.ar_routing import build_route_plan
from adapters.longlive_sparse.backends import _sequences
from adapters.longlive_sparse.grouped_staging import build_grouped_packing_recipe, pack_grouped_qkv


@pytest.mark.parametrize('batch,heads,exact', [(1, 2, 7), (2, 3, 0), (2, 1, 11)])
@pytest.mark.parametrize('strided', [False, True])
def test_packed_qkv_is_bitwise_identical_to_legacy(batch, heads, exact, strided):
    generator = torch.Generator().manual_seed(237)
    queries, history, dim = 19, 23, 16
    labels = torch.arange(queries).remainder(3).view(1,1,-1).expand(batch,heads,-1)
    selections = [[[torch.randperm(history, generator=generator)[:g+3].sort().values
                    for g in range(3)] for h in range(heads)] for b in range(batch)]
    frames = torch.zeros(batch,heads,history, dtype=torch.long)
    tokens = torch.arange(history).view(1,1,-1).expand(batch,heads,-1)
    plan = build_route_plan(method='test', routing_stage='pre-transfer', query_labels=labels,
        selections=selections, history_frame_ids=frames, history_token_ids=tokens,
        candidate_history_tokens=history, exact_k_tokens=exact, density=.5, metadata={})
    union = plan.union_frame_ids.shape[-1]
    def values(length):
        tensor = torch.randn(batch, length, heads, dim*(2 if strided else 1), generator=generator)
        return tensor[...,::2] if strided else tensor
    q, ek, ev, hk, hv = [values(length) for length in (queries,exact,exact,union,union)]
    recipe = build_grouped_packing_recipe(plan, exact_tokens=exact, union_tokens=union)
    packed = pack_grouped_qkv(recipe, q, ek, ev, hk, hv)
    old = _sequences(q, ek, ev, hk, hv, plan)
    for index in range(3):
        reference = torch.cat([item[3+index] for item in old]).unsqueeze(1)
        assert torch.equal(packed[index], reference)
    restored = torch.empty_like(q).contiguous().view(-1, dim)
    restored.index_copy_(0, packed[3], packed[0][:,0])
    assert torch.equal(restored.view(q.shape), q)
    assert recipe.metadata_bytes > 0
    with pytest.raises(ValueError):
        build_grouped_packing_recipe(plan, exact_tokens=exact+1, union_tokens=union)
