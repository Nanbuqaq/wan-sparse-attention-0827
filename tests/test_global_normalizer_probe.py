import math
import torch
from adapters.longlive_sparse.global_normalizer_probe import counterfactual_statistics


def test_known_zero_value_sink_requires_its_normalizer_not_its_value():
    q=torch.tensor([[[[1.,0.]]]],dtype=torch.float64)
    k=torch.tensor([[[[10*math.sqrt(2),0.]],[[0.,0.]]]],dtype=torch.float64)
    v=torch.tensor([[[[0.,0.]],[[1.,0.]]]],dtype=torch.float64)
    reference=torch.tensor([[[[1/(1+math.exp(10)),0.]]]],dtype=torch.float64)
    r=counterfactual_statistics(q,k,v,reference,1,1)
    assert r['reference_max_abs']<1e-12 and r['zero_global_V_relative_L2']==0
    assert r['global_mass_mean']>.999 and r['delete_global_KV_relative_L2']>10000
    assert r['zero_global_KV_relative_L2']>1000


def test_nonzero_global_values_have_an_independently_detected_contribution():
    q=torch.zeros(1,1,1,2,dtype=torch.float64);k=torch.zeros(1,2,1,2,dtype=torch.float64)
    v=torch.tensor([[[[2.,0.]],[[0.,0.]]]],dtype=torch.float64)
    reference=torch.tensor([[[[1.,0.]]]],dtype=torch.float64)
    r=counterfactual_statistics(q,k,v,reference,1,1)
    assert r['reference_max_abs']==0 and r['global_mass_mean']==.5
    assert r['zero_global_V_relative_L2']==1 and r['delete_global_KV_relative_L2']==1
