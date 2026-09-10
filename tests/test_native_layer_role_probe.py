import torch

from adapters.longlive_sparse.native_layer_role_probe import sampled_role_statistics


def test_compact_roles_reconstruct_full_FP32_attention_without_modifying_inputs():
    rng=torch.Generator().manual_seed(19)
    q=torch.randn(1,5,3,8,generator=rng);k=torch.randn(1,12,3,8,generator=rng);v=torch.randn(1,12,3,8,generator=rng)
    copies=[x.clone() for x in (q,k,v)];before=torch.backends.cuda.matmul.allow_tf32
    z,o=sampled_role_statistics(q,k,v,3)
    result=(z.softmax(-1)[...,None]*o).sum(-2)
    expected=torch.stack([(q[0,:,h]@k[0,:,h].T*(8**-.5)).softmax(-1)@v[0,:,h] for h in range(3)])
    assert torch.allclose(result,expected,atol=3e-7,rtol=1e-6)
    assert all(torch.equal(a,b) for a,b in zip(copies,(q,k,v)))
    assert torch.backends.cuda.matmul.allow_tf32==before


def test_group_statistics_retain_group_weight_and_direction_separately():
    q=torch.zeros(1,2,1,4);k=torch.zeros(1,8,1,4)
    v=torch.arange(32).float().reshape(1,8,1,4)
    z,o=sampled_role_statistics(q,k,v,2)
    assert torch.equal(z.softmax(-1),torch.full((1,2,4),.25))
    for group in range(4):assert torch.equal(o[0,:,group],v[0,2*group:2*group+2,0].mean(0).expand(2,-1))
