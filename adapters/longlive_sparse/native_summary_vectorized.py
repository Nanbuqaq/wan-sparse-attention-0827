"""Vectorize existing Block64 first moments; require exact GPU equality before use."""
import torch


def summarize_frame_vectorized(k,v,block_tokens=64):
    if k.shape!=v.shape or k.ndim!=3 or k.shape[0]<1 or block_tokens!=64:
        raise ValueError('matched nonempty native frame K/V and Block64 required')
    full,tail=divmod(k.shape[0],block_tokens);outputs=[]
    for value in (k,v):
        pieces=[]
        if full:pieces.append(value[:full*block_tokens].reshape(full,block_tokens,*value.shape[1:]).float().mean(1))
        if tail:pieces.append(value[full*block_tokens:].float().mean(0)[None])
        outputs.append(torch.cat(pieces) if len(pieces)>1 else pieces[0])
    counts=torch.tensor([block_tokens]*full+([tail] if tail else []),device=k.device)
    return *outputs,counts
