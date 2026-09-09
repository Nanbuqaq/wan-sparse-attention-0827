import math

import torch

from adapters.longlive_sparse.native_temporal_rephase import rephase_temporal_keys


def test_zero_delta_is_identity_and_spatial_channels_are_exact():
    key=torch.randn(1,8,2,128).bfloat16()
    assert rephase_temporal_keys(key,0) is key
    changed=rephase_temporal_keys(key,16)
    assert torch.equal(key[...,44:],changed[...,44:])
    assert not torch.equal(key[...,:44],changed[...,:44])


def test_first_pair_rotation_and_double_precision_inverse():
    key=torch.zeros(1,2,1,128,dtype=torch.float64);key[...,0]=1
    changed=rephase_temporal_keys(key,math.pi/2)
    assert torch.allclose(changed[...,0],torch.zeros_like(changed[...,0]),atol=1e-14)
    assert torch.allclose(changed[...,1],torch.ones_like(changed[...,1]),atol=1e-14)
    assert torch.allclose(rephase_temporal_keys(changed,-math.pi/2),key,atol=1e-14)


def test_common_phase_preserves_current_pair_but_not_historical_affinity():
    gen=torch.Generator().manual_seed(29)
    q=torch.randn(1,7,2,128,generator=gen,dtype=torch.float64)
    k=torch.randn(1,11,2,128,generator=gen,dtype=torch.float64)
    original=torch.einsum('bqhd,bkhd->bhqk',q,k)
    shifted_q=rephase_temporal_keys(q,8);shifted_k=rephase_temporal_keys(k,8)
    both=torch.einsum('bqhd,bkhd->bhqk',shifted_q,shifted_k)
    historical=torch.einsum('bqhd,bkhd->bhqk',shifted_q,k)
    assert torch.allclose(original,both,atol=1e-12,rtol=1e-12)
    assert not torch.allclose(original,historical)
