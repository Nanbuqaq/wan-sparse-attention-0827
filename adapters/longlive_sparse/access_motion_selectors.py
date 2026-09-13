"""Frozen diagnostic selectors over committed original Block64 KV groups.

These change the route, not the stored values. No teacher output is an input.
"""
import math
import torch
from .query_balanced_value import select_static_once


def select_recent_bridge(a, costs, budget, group_frames, frame_tokens):
    """Reserve floor(B/2/frame_tokens) recent whole eligible frames.

    Scores stay normalized over the original candidate set. The remaining
    groups use exactly the same static ranking as sum, with stable ties.
    """
    if len(group_frames)!=len(costs) or frame_tokens<=0:
        raise ValueError('invalid frame ownership geometry')
    frames=sorted(set(group_frames),reverse=True)
    for frame in frames:
        if sum(c for c,f in zip(costs,group_frames) if f==frame)!=frame_tokens:
            raise ValueError('reservation requires complete eligible frames')
    reserve=set(frames[:min(len(frames),budget//(2*frame_tokens))])
    ids=[i for i,f in enumerate(group_frames) if f not in reserve]
    mask=torch.tensor([f in reserve for f in group_frames],device=a.device).expand(a.shape[0],-1).clone()
    reserved=len(reserve)*frame_tokens
    if ids:
        selected,_,_=select_static_once(a[:,:,ids],[costs[i] for i in ids],budget-reserved)
        mask[:,ids]=selected
    cost=torch.tensor(costs,device=a.device)
    return mask,(a*mask[:,None]).sum(-1),(mask*cost).sum(-1)


def select_value_novelty(a, costs, budget, value_mean, epsilon=1e-6):
    """Exact sequential marginal novelty reference, bounded to <=512 groups.

    r=sum_q(a) is the frozen existing nonnegative relevance. Value summaries
    must be from past clean commits; no raw candidate V or residual is read.
    Working state is H*G*D plus H*G, not a G*G similarity matrix.
    """
    if a.ndim!=3 or not costs or a.shape[-1]!=len(costs) or len(costs)>512:
        raise ValueError('bounded H,Q,G input required')
    if min(costs)<=0 or budget<0 or value_mean.shape[:2]!=(len(costs),a.shape[0]):
        raise ValueError('invalid costs or per-head prototype geometry')
    vectors=value_mean.float().permute(1,0,2)
    norms=vectors.norm(dim=-1);valid=norms>=1e-6
    vectors=vectors/norms.clamp_min(1e-6)[...,None]
    cost=torch.tensor(costs,device=a.device,dtype=torch.long)
    relevance=a.sum(1).clamp_min(0)
    chosen=torch.zeros_like(relevance,dtype=torch.bool)
    novelty=valid.to(relevance.dtype)
    remaining=torch.full((a.shape[0],),budget,device=a.device,dtype=torch.long)
    seen=torch.zeros(a.shape[0],device=a.device,dtype=torch.bool)
    heads=torch.arange(a.shape[0],device=a.device)
    for _ in range(min(len(costs),budget//min(costs))):
        legal=(~chosen)&(cost[None]<=remaining[:,None])
        gain=relevance*(epsilon+novelty)
        ids=gain.masked_fill(~legal,-torch.inf).argmax(-1)
        take=legal.any(-1)
        chosen[heads,ids]|=take
        remaining-=cost[ids]*take
        selected_valid=valid[heads,ids]&take
        distance=((1-(vectors*vectors[heads,ids,None]).sum(-1).clamp(-1,1))/2)
        updated=torch.where(seen[:,None],torch.minimum(novelty,distance),distance)
        novelty=torch.where(selected_valid[:,None],updated,novelty)*valid
        seen|=selected_valid
    return chosen,(a*chosen[:,None]).sum(-1),budget-remaining


def stage_budgets(candidate_tokens, frame_tokens, mode):
    """Four actual denoise calls, same total optional pairs for equal Q shapes.

    Quarter/three-quarter allocations use whole frames; reject geometries that
    cannot realize the exact registered budget. Clean is unchanged externally.
    """
    if mode not in ('uniform','early_heavy','late_heavy'):
        raise ValueError('unknown allocation')
    if candidate_tokens%(4*frame_tokens):
        raise ValueError('exact quarter whole-frame quota unavailable')
    quarter=candidate_tokens//4
    values=[2*quarter]*4 if mode=='uniform' else [3*quarter,3*quarter,quarter,quarter]
    return list(reversed(values)) if mode=='late_heavy' else values
