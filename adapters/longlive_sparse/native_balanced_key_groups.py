"""Past-key-only balanced cosine bisection with a hard group-size cap."""
import math
import torch
from torch.nn.functional import normalize


@torch.inference_mode()
def balanced_key_groups(keys,*,max_tokens=64,iterations=6):
    if keys.ndim!=3 or keys.shape[0]<1 or min(max_tokens,iterations)<1:
        raise ValueError('nonempty token/head/channel keys and positive limits required')
    features=normalize(keys.float(),dim=-1).flatten(1)/math.sqrt(keys.shape[1])
    stack=[torch.arange(keys.shape[0],device=keys.device)];leaves=[]
    while stack:
        ids=stack.pop()
        if ids.numel()<=max_tokens:leaves.append(ids);continue
        x=features.index_select(0,ids)
        mean=normalize(x.mean(0),dim=0)
        first=x.index_select(0,(x@mean).argmin().reshape(1))[0]
        second=x.index_select(0,(x@first).argmin().reshape(1))[0]
        centers=torch.stack([first,second]);cut=(ids.numel()+1)//2
        for _ in range(iterations):
            margin=x@centers[0]-x@centers[1]
            order=margin.argsort(descending=True,stable=True)
            left,right=order[:cut],order[cut:]
            centers=normalize(torch.stack([x.index_select(0,left).mean(0),x.index_select(0,right).mean(0)]),dim=-1)
        stack.extend([ids.index_select(0,right),ids.index_select(0,left)])
    lengths=[v.numel() for v in leaves];flat=torch.cat(leaves).cpu().tolist();groups=[];offset=0
    for length in lengths:groups.append(sorted(flat[offset:offset+length]));offset+=length
    groups.sort(key=lambda g:g[0])
    if sorted(i for g in groups for i in g)!=list(range(keys.shape[0])):
        raise RuntimeError('key partition lost or duplicated an original source coordinate')
    return groups
