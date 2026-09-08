import torch

from adapters.longlive_sparse.native_episode_memory import admission_plan


def test_storage_does_not_enter_logical_episode_plan_identity():
    kwargs=dict(source_end=48,target_start=96,frames=8,frame_tokens=880,layers=30,heads=24,dim=128,dtype=torch.bfloat16)
    plan,sha=admission_plan(**kwargs)
    assert plan['source_frames']==list(range(40,48)) and plan['destination_token_range']==[0,7040]
    assert plan['position_policy']=='preserve_stored_absolute_RoPE'
    assert sha==admission_plan(**kwargs)[1]
    assert sha!=admission_plan(**dict(kwargs,source_end=96))[1]
