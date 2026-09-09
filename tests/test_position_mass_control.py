import torch

from scripts.analyze_position_mass_control import match_group_mass


def test_mass_control_is_exact_for_unchanged_within_group_values():
    rng=torch.Generator().manual_seed(5)
    z=torch.randn(2,3,4,generator=rng);v=torch.randn(2,3,4,8,generator=rng)
    result=match_group_mass(z,v,z[...,1]+1.3,v[...,1,:])
    assert torch.allclose(result['mass'],result['matched_mass'],atol=1e-7,rtol=0)
    assert torch.allclose(result['retimed'],result['mass_only'],atol=1e-7,rtol=0)


def test_equal_mass_does_not_imply_equal_readout():
    z=torch.zeros(1,1,4);v=torch.zeros(1,1,4,2)
    result=match_group_mass(z,v,z[...,1],torch.ones(1,1,2))
    assert torch.equal(result['mass'],result['matched_mass'])
    assert torch.equal(result['retimed']-result['mass_only'],torch.full((1,1,2),.25))
