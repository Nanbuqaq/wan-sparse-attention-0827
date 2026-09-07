"""Teacher-free raw-admission proxies for a summary-plus-raw representation.

Unlike relevance under deletion, these proxies approximate variation that a
mean prototype cannot represent. Diagonal moments and Q-group summaries are
heuristics, not guaranteed attention-error bounds. No Dense output/residual,
current full candidate KV, or future statistics enter this interface.
"""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class OnlineMomentContext:
    query_mean: torch.Tensor           # [B,H,G,D]
    query_second_moment: torch.Tensor  # same shape; E[Q^2], not E[Q]^2
    query_group_sizes: torch.Tensor    # [B,H,G]
    key_mean: torch.Tensor             # [B,H,P,D], committed fixed-RoPE groups
    key_diag_variance: torch.Tensor    # same shape
    value_mean: torch.Tensor           # same shape
    value_total_variance: torch.Tensor # [B,H,P], sum of diagonal V variances
    counts: torch.Tensor              # [B,H,P], group multiplicity


RISK_CANDIDATES=('mass_value','mass_key_variance','mass_kv_variation','mass_second_order',
                 'mass_q2_second_order','uniform_q2_second_order','peak_q2_second_order')


def score_raw_moment_risk(context, *, candidate, groups_per_block=4):
    if candidate not in RISK_CANDIDATES:raise ValueError('unknown moment-risk candidate')
    q=context.query_mean; k=context.key_mean; v=context.value_mean
    if q.ndim!=4 or k.ndim!=4 or q.shape[:2]!=k.shape[:2] or q.shape[-1]!=k.shape[-1] or v.shape!=k.shape:
        raise ValueError('invalid compact summary geometry')
    if (context.query_second_moment.shape!=q.shape or context.query_group_sizes.shape!=q.shape[:-1]
            or context.key_diag_variance.shape!=k.shape or context.value_total_variance.shape!=k.shape[:-1]
            or context.counts.shape!=k.shape[:-1] or k.shape[2]%groups_per_block):
        raise ValueError('moment shape mismatch')
    if bool((context.counts.sum(-1)<=0).any()) or bool((context.query_group_sizes.sum(-1)<=0).any()):
        raise ValueError('nonempty key and query groups required')
    logits=torch.einsum('bhgd,bhpd->bhgp',q.float(),k.float())/math.sqrt(q.shape[-1])
    logits=logits+context.counts.float().log()[:,:,None]
    mass=logits.softmax(-1)
    use_q2='q2' in candidate
    q_energy=context.query_second_moment.float() if use_q2 else q.float().square()
    variance=torch.einsum('bhgd,bhpd->bhgp',q_energy,context.key_diag_variance.float().clamp_min(0))/q.shape[-1]
    variance=variance.clamp_min(0)
    vnorm=v.float().norm(dim=-1)[:,:,None]
    vdispersion=context.value_total_variance.float().clamp_min(0).sqrt()[:,:,None]
    variation=variance.sqrt()*vdispersion
    if candidate=='mass_value':risk=mass*vnorm
    elif candidate=='mass_key_variance':risk=mass*variance
    elif candidate=='mass_kv_variation':risk=mass*variation
    else:
        # Small-variation numerator/normalization-inspired proxy; not a bound.
        base=variation+.5*variance*vnorm
        risk=base if candidate=='uniform_q2_second_order' else mass*base
    risk=risk.masked_fill(context.counts[:,:,None]==0,0.)
    risk=risk.reshape(*risk.shape[:-1],-1,groups_per_block).sum(-1)
    active=context.query_group_sizes>0
    if candidate=='peak_q2_second_order':
        return risk.masked_fill(~active[...,None],0.).amax(2)
    weights=context.query_group_sizes.float()/context.query_group_sizes.sum(-1,keepdim=True)
    return (risk*weights[...,None]).sum(2)
