import itertools
import numpy as np
import pytest
import torch
from adapters.longlive_sparse.query_balanced_value import select_static_once,select_reference,select_batched


def test_all_short_cost_combinations_and_budget_edges():
    rng=np.random.default_rng(61010)
    for length in range(1,7):
        for costs in itertools.product((48,64),repeat=length):
            random=rng.random((3,4,length)).astype('float32')
            random[1]=0;random[2]=np.arange(length)[None]%2
            a=torch.tensor(random)
            for budget in sorted(set([0,1,47,48,49,63,64,65,sum(costs)-1,sum(costs),sum(costs)+1,*range(0,sum(costs),16)])):
                mask,coverage,used=select_static_once(a,costs,budget)
                for head in range(3):
                    ids,c,u=select_reference(random[head],costs,budget,balanced=False)
                    assert mask[head].nonzero().flatten().tolist()==ids
                    assert used[head]==u
                    np.testing.assert_allclose(coverage[head],c,atol=2e-6)


def test_tail_and_old_tensor_reference():
    a=torch.tensor([[[9.,8.,7.,6.]],[[1.,2.,3.,4.]]])
    costs=[64,64,48,48]
    for budget in (48,64,112,128,160,224):
        mask,_,used=select_static_once(a,costs,budget)
        old,_,old_used=select_batched(a,costs,budget,balanced=False)
        assert torch.equal(mask,old) and torch.equal(used,old_used)
    assert select_static_once(a,costs,112)[0][0].tolist()==[True,False,True,False]
    with pytest.raises(ValueError,match='only48/64'):
        select_static_once(a,[32,64,48,48],112)
