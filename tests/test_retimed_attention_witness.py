import torch

from adapters.longlive_sparse.native_temporal_rephase import rephase_temporal_keys
from scripts.analyze_retimed_attention import input_witness


def test_first_layer_source_only_rotation_is_distinguished_from_later_trajectory_changes():
    g=torch.Generator().manual_seed(20260909)
    old=dict(frame_tokens=1,q=torch.randn(1,4,2,128,generator=g).bfloat16(),
        k=torch.randn(1,32,2,128,generator=g).bfloat16(),v=torch.randn(1,32,2,128,generator=g).bfloat16())
    new={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in old.items()}
    new['k'][:,8:16]=rephase_temporal_keys(old['k'][:,8:16],64)
    assert all(input_witness(old,new,64).values())
    new['q']+=1;new['k'][:,24:]+=1;new['v'][:,24:]+=1
    witness=input_witness(old,new,64)
    assert not witness['Q_bitwise_equal'] and not witness['all_V_bitwise_equal'] and not witness['all_other_K_bitwise_equal']
    assert witness['source_V_bitwise_equal'] and witness['source_K_equals_declared_temporal_rotation']
