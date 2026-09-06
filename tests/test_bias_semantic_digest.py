from dataclasses import replace
import pytest
import torch

from adapters.longlive_sparse.attention_bias import AttentionBiasPlan


def test_executed_context_weight_changes_bias_sha_not_just_metadata_label():
    plan = AttentionBiasPlan(('identity', 'scene'), torch.tensor([[[1., 0.]]]),
        torch.tensor([[[[0., 1.]]]]), torch.ones(1, 1, 1), metadata={'context_weight': .2})
    assert plan.digest() != replace(plan, metadata={'context_weight': .8}).digest()
    assert plan.digest() == replace(plan, metadata={'context_weight': .2, 'timing_s': 9.}).digest()
    with pytest.raises(ValueError, match='finite'):
        replace(plan, metadata={'context_weight': float('nan')})
