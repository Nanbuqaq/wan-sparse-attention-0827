import torch
from adapters.longlive_sparse import selectors


def test_full_local_density_does_not_compute_or_sort_scores(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('100% local baseline must not pay sparse routing')
    monkeypatch.setattr(selectors,'_query_block_means',forbidden)
    q=torch.randn(2,7,3,8);k=torch.randn(2,11,3,8)
    indices=selectors.select_block64_from_tensor(q,k,1.,4)
    assert torch.equal(indices,torch.arange(11).view(1,1,-1).expand(2,3,-1))
