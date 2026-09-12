import numpy as np
import pytest
import torch

from adapters.longlive_sparse.query_balanced_value import select_batched, select_reference


@pytest.mark.parametrize('balanced', [False, True])
def test_single_group_batches_match_independent_cpu_reference(balanced):
    rng = np.random.default_rng(932)
    x = rng.random((3, 7, 19)).astype('float32')
    x /= x.sum(-1, keepdims=True)
    costs = [48 if i % 4 == 0 else 64 for i in range(19)]
    for budget in (0, 47, 48, 127, 432, sum(costs)):
        mask, coverage, used = select_batched(torch.tensor(x), costs, budget,
                                               balanced=balanced, batch_size=1)
        for h in range(3):
            ids, c, u = select_reference(x[h], costs, budget, balanced=balanced)
            assert mask[h].nonzero().flatten().tolist() == ids
            np.testing.assert_allclose(coverage[h].numpy(), c, atol=2e-7)
            assert used[h] == u


def test_batched_budget_coverage_and_no_affordable_group_left():
    torch.manual_seed(55)
    a = torch.rand(4, 128, 112)
    a /= a.sum(-1, keepdim=True)
    costs = [48 if i % 14 == 13 else 64 for i in range(112)]
    for budget in (0, 49, 200, 3520, 7000):
        mask, coverage, used = select_batched(a, costs, budget, balanced=True)
        assert (used <= budget).all()
        assert torch.equal(used, (mask*torch.tensor(costs)).sum(-1))
        torch.testing.assert_close(coverage, (a*mask[:, None]).sum(-1))
        assert not ((~mask) & (torch.tensor(costs)[None] <= (budget-used)[:, None])).any()


def test_stable_ties_and_explicit_geometry_gate():
    a = torch.ones(1, 3, 8)/8
    mask, _, _ = select_batched(a, [64]*8, 256, balanced=True)
    assert mask.nonzero()[:, 1].tolist() == [0, 1, 2, 3]
    with pytest.raises(ValueError, match='bounded'):
        select_batched(torch.ones(1, 129, 8), [64]*8, 256, balanced=True)


def test_closed_loop_batch_has_two_matched_arms_per_valid_task(tmp_path):
    import json
    from pathlib import Path
    from scripts.run_native_duration_wave import build_wave2_cases
    spec=json.loads((Path(__file__).resolve().parents[1]/'configs/system/wave2_scenarios.json').read_text())
    valid=['w2_rotating_wooden_bird','w2_tracking_delivery_cart']
    cases=build_wave2_cases(spec,'query_balance',tmp_path,tmp_path,tmp_path,20261010,valid,'fixed-noise')
    assert len(cases)==4 and len({c['id'] for c in cases})==4
    for scenario in valid:
        pair=[c for c in cases if c['scenario']==scenario]
        assert {c['selector'] for c in pair}=={'query_sum_batch4','query_balanced_batch4'}
        for c in pair:
            assert '--wave2-capture' not in c['cmd']
            assert c['cmd'][c['cmd'].index('--expected-noise-sha256')+1]=='fixed-noise'
    with pytest.raises(ValueError,match='valid continuous'):
        build_wave2_cases(spec,'query_balance',tmp_path,tmp_path,tmp_path,20261010,['w2_ceramic_jug_revisit'])
