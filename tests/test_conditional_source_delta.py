import pytest
import torch
from adapters.longlive_sparse.conditional_source_delta import apply_condition_delta


def test_null_is_exact_even_with_disparate_magnitudes():
    raw=torch.tensor([.001,1.,1000.,-1e5],dtype=torch.bfloat16)
    past=torch.tensor([1e9,-1e7,1e-6,1e-8],dtype=torch.bfloat16)
    for sign in (-1,1):assert torch.equal(apply_condition_delta(raw,past,past,sign),raw)


def test_direction_is_frozen_and_no_norm_rescaling_or_clipping():
    raw=torch.tensor([1.,2.,-3.],dtype=torch.bfloat16)
    past=torch.tensor([3.,3.,3.],dtype=torch.bfloat16);current=torch.tensor([4.,1.,8.],dtype=torch.bfloat16)
    assert torch.equal(apply_condition_delta(raw,current,past,1),torch.tensor([2.,0.,2.],dtype=torch.bfloat16))
    assert torch.equal(apply_condition_delta(raw,current,past,-1),torch.tensor([0.,4.,-8.],dtype=torch.bfloat16))
    assert not torch.equal(apply_condition_delta(raw,current,past,1),raw)
    with pytest.raises(ValueError):apply_condition_delta(raw,current,past,.5)
    with pytest.raises(ValueError):apply_condition_delta(raw,current[:-1],past,1)
