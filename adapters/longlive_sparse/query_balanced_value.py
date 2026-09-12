"""Bounded per-head query coverage reference on fixed original-token groups."""
import math
import numpy as np
import torch


def stratified_sites(frames,height,width,grid=4):
    ys=torch.linspace(0,height-1,grid).round().long();xs=torch.linspace(0,width-1,grid).round().long()
    return torch.tensor([t*height*width+int(y)*width+int(x) for t in range(frames) for y in ys for x in xs])


def normalized_values(query,key_mean,value_mean,counts):
    logits=torch.einsum('qhd,bhd->hqb',query.float(),key_mean.float())/math.sqrt(query.shape[-1])
    p=(logits+counts.float().log()[None,None]).softmax(-1)
    r=p*value_mean.float().norm(dim=-1).T[:,None,:]
    return r/r.sum(-1,keepdim=True).clamp_min(1e-12)


def select_reference(a,counts,budget,*,balanced,cap=.8):
    """CPU reference only. Stable ties; original groups remain indivisible."""
    if not 0<cap<=1 or budget<0:raise ValueError('invalid registered coverage/budget')
    x=np.asarray(a,dtype=np.float32);cost=np.asarray(counts,dtype=np.int64)
    if x.ndim!=2 or x.shape[1]!=len(cost) or np.any(cost<=0):raise ValueError('group geometry differs')
    selected=[];coverage=np.zeros(x.shape[0],np.float32);remaining=budget
    if not balanced:
        for b in np.argsort(-x.sum(0),kind='stable'):
            if cost[b]<=remaining:selected.append(int(b));coverage+=x[:,b];remaining-=int(cost[b])
    else:
        available=np.ones(len(cost),bool)
        for _ in range(len(cost)):
            legal=available&(cost<=remaining)
            if not legal.any():break
            gain=(np.minimum(cap,coverage[:,None]+x)-np.minimum(cap,coverage[:,None])).sum(0)/cost
            gain[~legal]=-np.inf;b=int(np.argmax(gain))
            selected.append(b);coverage+=x[:,b];remaining-=int(cost[b]);available[b]=False
    return sorted(selected),coverage,budget-remaining


def select_batched(a, costs, budget, *, balanced, batch_size=4, cap=.8):
    """Deterministic tensor batched-greedy variant; not exact sequential greedy.

    Costs are immutable CPU metadata. Heads have independent original-token
    budgets. No tensor-to-host operation occurs inside the selection loop.
    A batch uses stale marginal utilities within that batch. Stable ties prefer
    lower group IDs. Scratch is H*Q*G, never H*Q*G*D.
    """
    if a.ndim != 3 or a.shape[-1] != len(costs) or not costs:
        raise ValueError('expected nonempty H,Q,G and matching costs')
    if min(costs) <= 0 or budget < 0 or batch_size < 1 or not 0 < cap <= 1:
        raise ValueError('invalid cost, budget or coverage')
    if a.shape[0] > 24 or a.shape[1] > 128 or a.shape[2] > 512:
        raise ValueError('outside bounded Wave2 selector geometry')
    cost = torch.tensor(costs, device=a.device, dtype=torch.long)
    chosen = torch.zeros((a.shape[0], len(costs)), device=a.device, dtype=torch.bool)
    remaining = torch.full((a.shape[0],), budget, device=a.device, dtype=torch.long)
    coverage = torch.zeros(a.shape[:2], device=a.device, dtype=a.dtype)
    static = a.sum(1)
    # Full batches consume at least batch_size*mincost. Once a batch cannot
    # fit, fewer than batch_size*maxcost tokens remain. The conservative tail
    # also covers variable costs and heads that exhaust at different times.
    rounds = min(len(costs), math.ceil(budget/(batch_size*min(costs)))
                 + math.ceil(batch_size*max(costs)/min(costs)))
    for _ in range(rounds):
        gain = (torch.minimum(a, (cap-coverage).clamp_min(0).unsqueeze(-1)).sum(1)
                / cost if balanced else static)
        legal = ~chosen & (cost[None] <= remaining[:, None])
        order = gain.masked_fill(~legal, -torch.inf).argsort(dim=-1, descending=True, stable=True)
        ids = order[:, :min(batch_size, len(costs))]
        sizes = cost[ids]
        take = legal.gather(1, ids) & (sizes.cumsum(1) <= remaining[:, None])
        addition = torch.zeros_like(chosen).scatter(1, ids, take)
        chosen |= addition
        remaining -= (sizes*take).sum(1)
        coverage += (a*addition[:, None]).sum(-1)
    return chosen, coverage, budget-remaining


def execute_per_head(q, k, v, chosen, group_for_token, max_selected_k):
    """Execute original KV with independent head routes using native FA2 varlen.

    group_for_token is a device int64 array: -1 denotes mandatory context.
    nonzero's dynamic-size synchronization, pack and layout costs are included
    by the caller. No physical union is used as a substitute for head routes.
    """
    from flash_attn import flash_attn_varlen_func
    if q.shape[0] != 1 or k.shape != v.shape or q.dtype != torch.bfloat16:
        raise ValueError('expected one BF16 native window')
    heads, length, dim = q.shape[2], q.shape[1], q.shape[3]
    visible = chosen[:, group_for_token.clamp_min(0)] | (group_for_token[None] < 0)
    coordinates = visible.nonzero(as_tuple=False)
    hk = k[0].permute(1, 0, 2)[coordinates[:, 0], coordinates[:, 1]][:, None]
    hv = v[0].permute(1, 0, 2)[coordinates[:, 0], coordinates[:, 1]][:, None]
    hq = q[0].permute(1, 0, 2).contiguous().reshape(heads*length, 1, dim)
    cuq = torch.arange(heads+1, device=q.device, dtype=torch.int32)*length
    cuk = torch.cat([torch.zeros(1, device=q.device, dtype=torch.int32),
                     visible.sum(-1, dtype=torch.int32).cumsum(0, dtype=torch.int32)])
    output = flash_attn_varlen_func(hq, hk, hv, cuq, cuk, length, max_selected_k,
                                    dropout_p=0.0, causal=False)
    return output.reshape(heads, length, dim).permute(1, 0, 2)[None].contiguous(), visible
