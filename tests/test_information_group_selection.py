import itertools
import torch

from adapters.longlive_sparse.information_group_selection import ExactTwoCostBudget,rank_discount,IncrementalValueGroups


def test_exact_two_cost_solver_matches_exhaustive_same_budget_objective():
    costs=[48,48,48,48,64,64,64,64]
    scores=torch.tensor([[9.,2.,7.,3.,8.,4.,5.,1.],[1.,7.,2.,8.,4.,3.,6.,5.]])
    mask=ExactTwoCostBudget(costs,192,'cpu').select(scores)
    assert (mask*torch.tensor(costs)).sum(1).tolist()==[192,192]
    feasible=[b for b in itertools.product((0,1),repeat=len(costs)) if sum(c*x for c,x in zip(costs,b))==192]
    for h in range(2):
        best=max(sum(float(scores[h,i])*b[i] for i in range(len(costs))) for b in feasible)
        assert float((scores[h]*mask[h]).sum())==best


def test_rank_discount_is_head_specific_and_stable_for_exact_ties():
    scores=torch.tensor([[4.,4.,2.,1.],[1.,2.,4.,4.]])
    groups=torch.tensor([[0,0,1,1],[0,1,0,1]])
    actual=rank_discount(scores,groups,2)
    torch.testing.assert_close(actual,torch.tensor([[4.,2.,2.,.5],[.5,1.,4.,4.]]))


def test_incremental_membership_keeps_survivors_and_expires_old_versions():
    torch.manual_seed(7);values=torch.randn(24,2,8);counts=torch.full((24,),64)
    groups=IncrementalValueGroups();keys=[('v1',i) for i in range(24)]
    old=groups.update(0,keys,values,counts).clone()
    assert groups.update(0,keys,values,counts).data_ptr()==groups.states[0]['groups'].data_ptr()
    new_keys=keys[8:]+[('v2',i) for i in range(8)]
    new_values=torch.cat([values[8:],torch.randn(8,2,8)])
    new=groups.update(0,new_keys,new_values,counts)
    assert torch.equal(new[:,:16],old[:,8:])
    assert int(new.max())<16 and groups.expired_blocks==8 and groups.retained_blocks==16
    assert groups.hits==1 and len(groups.states[0]['keys'])==24


def test_information_cohort_preserves_controls_and_late_native_pairing(tmp_path):
    import json
    from pathlib import Path
    from scripts.run_native_duration_wave import build_wave2_cases,case_lane_indices
    spec=json.loads((Path(__file__).resolve().parents[1]/'configs/system/wave2_scenarios.json').read_text())
    rows=build_wave2_cases(spec,'information_groups',tmp_path,tmp_path,tmp_path,20261010)
    lanes=case_lane_indices(rows,2)
    assert len(rows)==12 and list(map(len,lanes))==[6,6]
    for indices in lanes:
        assert len({rows[i]['scenario'] for i in indices})==1
        assert rows[indices[-1]]['method']=='native_late'
        assert all('--wave2-query-groups' not in rows[i]['cmd'] for i in indices)
