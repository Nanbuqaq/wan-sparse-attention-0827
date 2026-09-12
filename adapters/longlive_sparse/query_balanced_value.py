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
