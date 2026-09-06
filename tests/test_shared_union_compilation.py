import torch
import pytest
from adapters.longlive_sparse.ar_routing import build_route_plan


@pytest.mark.parametrize('batch,heads,groups,tokens',[(1,2,3,17),(2,3,7,41),(1,12,74,4680)])
def test_shared_union_precompilation_preserves_every_plan_field(batch,heads,groups,tokens):
    labels=(torch.arange(tokens)%groups).view(1,1,-1).expand(batch,heads,-1).clone()
    frames=torch.arange(3).repeat_interleave(128).view(1,1,-1).expand(batch,heads,-1)
    coords=torch.arange(128).repeat(3).view(1,1,-1).expand_as(frames)
    rows=[[torch.randperm(384)[:96].sort().values for _ in range(heads)] for _ in range(batch)]
    kwargs=dict(method='test',routing_stage='pre-transfer',query_labels=labels,history_frame_ids=frames,
        history_token_ids=coords,candidate_history_tokens=384,exact_k_tokens=128,density=.25,metadata={'audit':True})
    old=build_route_plan(**kwargs,selections=[[[r for _ in range(groups)] for r in headrows] for headrows in rows])
    new=build_route_plan(**kwargs,selections=[[[r] for r in headrows] for headrows in rows],shared_union=True)
    assert old.digest()==new.digest()
    for name,value in old.state_dict().items():
        other=new.state_dict()[name]
        assert torch.equal(value,other) if isinstance(value,torch.Tensor) else value==other
    assert torch.equal(labels,(torch.arange(tokens)%groups).view(1,1,-1).expand(batch,heads,-1))


def test_shared_union_does_not_accept_distinct_group_rows():
    with pytest.raises(ValueError,match='one row'):
        build_route_plan(method='x',routing_stage='pre-transfer',query_labels=torch.tensor([[[0,1]]]),
            selections=[[[torch.tensor([0]),torch.tensor([1])]]],history_frame_ids=torch.tensor([[[0,0]]]),
            history_token_ids=torch.tensor([[[0,1]]]),candidate_history_tokens=2,exact_k_tokens=2,
            density=.5,metadata={},shared_union=True)
