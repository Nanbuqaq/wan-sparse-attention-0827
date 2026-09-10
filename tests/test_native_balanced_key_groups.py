import torch
from adapters.longlive_sparse.native_balanced_key_groups import balanced_key_groups


def test_coverage_hard_cap_and_rng_input_ownership():
    generator=torch.Generator().manual_seed(91);keys=torch.randn(129,2,4,generator=generator)
    original=keys.clone();state=torch.get_rng_state().clone();groups=balanced_key_groups(keys)
    assert sorted(i for g in groups for i in g)==list(range(129))
    assert all(0<len(g)<=64 for g in groups)
    assert torch.equal(keys,original) and torch.equal(torch.get_rng_state(),state)
    assert groups==balanced_key_groups(keys)


def test_directionally_separated_populations_get_separate_groups():
    keys=torch.zeros(128,2,4);keys[:64,:,0]=1;keys[64:,:,0]=-1
    groups=balanced_key_groups(keys)
    assert groups==[list(range(64)),list(range(64,128))]
