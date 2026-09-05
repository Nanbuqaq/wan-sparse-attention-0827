import torch
import pytest
from adapters.longlive_sparse.offline_eval import output_error_metrics


def test_large_identical_bf16_has_valid_cosine_distance():
    values=torch.randn(1000000,generator=torch.Generator().manual_seed(9)).bfloat16()
    result=output_error_metrics(values,values.clone())
    assert result['relative_l2']==result['max_abs']==0.
    assert 0.<=result['one_minus_cosine']<1e-12


def test_zero_and_nonfinite_output_metric_contract():
    assert output_error_metrics(torch.zeros(4),torch.zeros(4))['one_minus_cosine']==0.
    assert output_error_metrics(torch.ones(4),-torch.ones(4))['one_minus_cosine']==2.
    with pytest.raises(ValueError,match='finite'):
        output_error_metrics(torch.ones(4),torch.full((4,),float('nan')))
