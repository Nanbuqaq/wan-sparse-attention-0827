import torch

from scripts.analyze_native_attention_teacher import decompose


def record():
    generator=torch.Generator().manual_seed(17)
    q=torch.randn(1,4,2,8,generator=generator).bfloat16()
    k=torch.randn(1,48,2,8,generator=generator).bfloat16()
    v=torch.randn(1,48,2,8,generator=generator).bfloat16()
    scores=torch.einsum('bqhd,bkhd->bhqk',q.float(),k.float())*8**-.5
    output=torch.einsum('bhqk,bkhd->bqhd',scores.softmax(-1),v.float()).bfloat16()
    return dict(q=q,k=k,v=v,native_output=output,frame_tokens=2,softmax_scale=8**-.5,
        query_frame=96,phase=0,layer=0,full_Q_tokens=4,query_sites=['center','left','right','lower'])


def test_group_logsumexp_matches_full_attention_and_gain_one_is_identity():
    r=record();report,data=decompose(r,['initial','source','current'])
    q=r['q'][0].permute(1,0,2).float();k=r['k'][0].permute(1,0,2).float();v=r['v'][0].permute(1,0,2).float()
    expected=torch.bmm((torch.bmm(q,k.transpose(1,2))*r['softmax_scale']).softmax(-1),v)
    assert torch.allclose(expected,data['fp32_output'],atol=1e-6,rtol=1e-6)
    assert torch.allclose(data['mass'].sum(-1),torch.ones_like(data['mass'][...,0]))
    assert report['FP32_replay_gate']
    gains=report['offline_source_gain_sensitivity_not_quality']
    assert gains['1.0']['relative_output_change']['max']==0
    assert gains['2.0']['source_mass']['mean']>gains['1.0']['source_mass']['mean']


def test_remove_group_does_not_use_cancellation_of_tiny_global_mass():
    r=record();r['q'].fill_(10);r['k'][:,:16].fill_(10);r['k'][:,16:].fill_(-10)
    report,data=decompose(r,['initial','source','current'])
    assert torch.isfinite(data['role_outputs']).all()
    assert all(torch.isfinite(torch.tensor(v['remove_role_output_sensitivity']['mean'])) for v in report['roles'].values())
