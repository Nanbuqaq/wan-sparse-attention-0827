"""Fixed source set log-beta bias via two disjoint native FA2 partials.

At dropout0 the installed FA2 wrapper returns LSE without materializing Q*K
probabilities. Partial BF16 outputs introduce rounding: beta1 has its own
numerical and closed-loop control, never claimed bitwise native-equivalent.
"""
import math
import torch


def weighted_source_attention(q,k,v,source_indices,other_indices,beta):
    from flash_attn import flash_attn_func
    if beta not in (.5,1.,2.):raise ValueError('only frozen beta0.5/1/2 diagnostic')
    if source_indices.numel()+other_indices.numel()!=k.shape[1]:raise ValueError('partition must cover permitted graph')
    if not source_indices.numel() or not other_indices.numel():raise ValueError('both partitions must be nonempty')
    outputs=[];lses=[]
    for indices in (source_indices,other_indices):
        output,lse,probability=flash_attn_func(q,k.index_select(1,indices),v.index_select(1,indices),
            dropout_p=0.,causal=False,return_attn_probs=True)
        if probability.numel():raise RuntimeError('unexpected materialized attention matrix')
        outputs.append(output);lses.append(lse)
    weight=(lses[0]+math.log(beta)-lses[1]).sigmoid()
    blend=weight.transpose(1,2).unsqueeze(-1)
    output=(blend*outputs[0].float()+(1-blend)*outputs[1].float()).to(q.dtype)
    return output,weight


def independent_sample_error(q,k,v,source_indices,beta,output):
    """Isolated FP32 teacher on32 fixed Q sites, never used for routing/weight."""
    sites=torch.linspace(0,q.shape[1]-1,32,device=q.device).round().long()
    logits=torch.einsum('qhd,khd->hqk',q[0,sites].float(),k[0].float())/math.sqrt(q.shape[-1])
    logits[:,:,source_indices]+=math.log(beta)
    teacher=torch.einsum('hqk,khd->qhd',logits.softmax(-1),v[0].float())
    error=(output[0,sites].float()-teacher).square().sum()/teacher.square().sum().clamp_min(1e-30)
    return float(error.sqrt())
