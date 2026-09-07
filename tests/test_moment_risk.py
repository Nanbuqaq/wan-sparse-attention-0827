from dataclasses import replace
import pytest
import torch
from adapters.longlive_sparse.moment_risk import OnlineMomentContext,score_raw_moment_risk,RISK_CANDIDATES


def make_context():
    return OnlineMomentContext(torch.ones(1,2,3,8),torch.ones(1,2,3,8),torch.ones(1,2,3),
        torch.ones(1,2,8,8),torch.ones(1,2,8,8),torch.ones(1,2,8,8),torch.ones(1,2,8),torch.ones(1,2,8)*16)


@pytest.mark.parametrize('candidate',RISK_CANDIDATES)
def test_shape_and_finite_risk(candidate):
    context=make_context();score=score_raw_moment_risk(context,candidate=candidate)
    assert score.shape==(1,2,2) and torch.isfinite(score).all() and (score>=0).all()


def test_zero_key_variation_has_zero_moment_risk():
    context=make_context();context=replace(context,key_diag_variance=torch.zeros_like(context.key_diag_variance))
    assert torch.equal(score_raw_moment_risk(context,candidate='mass_second_order'),torch.zeros(1,2,2))


def test_query_second_moment_retains_variation_when_query_mean_cancels():
    context=make_context();context=replace(context,query_mean=torch.zeros_like(context.query_mean))
    assert score_raw_moment_risk(context,candidate='mass_second_order').sum()==0
    assert score_raw_moment_risk(context,candidate='mass_q2_second_order').sum()>0


def test_empty_prototype_slots_do_not_contribute():
    context=make_context();context.counts[:,:,4:]=0
    assert torch.equal(score_raw_moment_risk(context,candidate='mass_q2_second_order')[:,:,1],torch.zeros(1,2))
