import pytest
import torch
from adapters.longlive_sparse.access_motion_selectors import select_recent_bridge,select_value_novelty,stage_budgets
from adapters.longlive_sparse.query_balanced_value import select_static_once


def test_bridge_reserves_recent_without_renormalizing_or_expanding_budget():
    a=torch.tensor([[[10.,9.,8.,7.,6.,5.,4.,3.]]])
    costs=[64]*8;frames=[0,0,1,1,2,2,3,3]
    mask,coverage,used=select_recent_bridge(a,costs,256,frames,128)
    assert mask.tolist()==[[True,True,False,False,False,False,True,True]]
    assert used.item()==256 and coverage.item()==26
    zero=select_recent_bridge(a,costs,64,frames,128)
    ref=select_static_once(a,costs,64)
    assert all(torch.equal(x,y) for x,y in zip(zero,ref))
    with pytest.raises(ValueError,match='complete'):
        select_recent_bridge(a,costs,256,frames,880)


def test_novelty_rejects_duplicate_direction_then_uses_opposite():
    a=torch.tensor([[[4.,3.,2.,1.]]])
    values=torch.tensor([[[1.,0.]],[[1.,0.]], [[-1.,0.]],[[0.,0.]]])
    mask,_,used=select_value_novelty(a,[64]*4,128,values)
    assert mask.tolist()==[[True,False,True,False]] and used.item()==128
    assert select_static_once(a,[64]*4,128)[0].tolist()==[[True,True,False,False]]


def test_novelty_zero_norm_stable_ties_and_indivisible_budget():
    a=torch.ones(2,3,4);values=torch.zeros(4,2,2)
    mask,_,used=select_value_novelty(a,[64,48,64,48],112,values)
    assert mask.tolist()==[[True,True,False,False]]*2
    assert used.tolist()==[112,112]
    assert not select_value_novelty(a,[64]*4,0,values)[0].any()


def test_stage_quota_is_exact_and_not_a_percentage_average():
    for mode in ('uniform','early_heavy','late_heavy'):
        b=stage_budgets(16*880,880,mode)
        assert sum(b)==2*16*880 and all(x%880==0 for x in b)
    assert stage_budgets(16*880,880,'early_heavy')==list(reversed(stage_budgets(16*880,880,'late_heavy')))
    with pytest.raises(ValueError):stage_budgets(3*880,880,'uniform')
