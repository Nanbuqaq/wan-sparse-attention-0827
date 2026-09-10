"""Candidate source value relative to compact summaries of the kept context.

This estimates a joint denominator/reference, not exact deletion error. Inputs
are current Q and small source/protected K/V means, never a teacher output.
"""
import math
import torch


def source_context_scores(q,source_k,source_v,source_counts,kept_k,kept_v,kept_counts,policy,samples=32):
    if policy not in ('mass_value','contrast_value'):raise ValueError('unknown value proxy')
    if (q.ndim!=3 or source_k.shape!=source_v.shape or kept_k.shape!=kept_v.shape
        or source_k.shape[1:]!=q.shape[1:] or kept_k.shape[1:]!=q.shape[1:]
        or source_counts.shape!=(source_k.shape[0],) or kept_counts.shape!=(kept_k.shape[0],)):
        raise ValueError('summary geometry mismatch')
    sites=torch.linspace(0,q.shape[0]-1,min(q.shape[0],samples),device=q.device).round().long()
    qq=q.index_select(0,sites).float()
    key=torch.cat([source_k,kept_k]).float();value=torch.cat([source_v,kept_v]).float().permute(1,0,2)
    counts=torch.cat([source_counts,kept_counts]).float()
    mass=(torch.einsum('qhd,ghd->hqg',qq,key)/math.sqrt(q.shape[-1])+counts.log()[None,None,:]).softmax(-1)
    source_mass=mass[:,:,:source_k.shape[0]]
    if policy=='mass_value':score=source_mass*source_v.float().permute(1,0,2).norm(dim=-1)[:,None,:]
    else:
        reference=torch.einsum('hqg,hgd->hqd',mass,value)
        score=source_mass*(source_v.float().permute(1,0,2)[:,None,:,:]-reference[:,:,None,:]).norm(dim=-1)
    return score.mean(1)
