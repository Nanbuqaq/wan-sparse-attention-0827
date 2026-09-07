"""Index-time within-block information grouping for prototype-tail analysis.

Groups use committed K features only, not current/future queries or outputs.
This is a representation probe, not a promoted online storage implementation.
"""
import torch
import torch.nn.functional as F
from .prototype_tail import PrototypeTail


def build_feature_tail(key, value, selected_indices, *, frame_tokens, block_tokens=64,
                       groups=4, grouping='key_kmeans', iterations=4, seed=20260907):
    if grouping not in ('spatial_groups', 'random_groups', 'key_kmeans'):
        raise ValueError('unknown prototype grouping')
    if key.shape != value.shape or key.ndim != 4 or key.shape[1] % frame_tokens:
        raise ValueError('complete frame-major candidate tensors required')
    batch, tokens, heads, dim = key.shape
    frames = tokens//frame_tokens
    blocks = (frame_tokens+block_tokens-1)//block_tokens
    pad = blocks*block_tokens-frame_tokens
    count_blocks = batch*heads*frames*blocks
    def block_view(tensor):
        tensor = tensor.permute(0,2,1,3).float().reshape(batch,heads,frames,frame_tokens,dim)
        return F.pad(tensor,(0,0,0,pad)).reshape(count_blocks,block_tokens,dim)
    kb, vb = block_view(key), block_view(value)
    valid = F.pad(torch.ones(batch,heads,frames,frame_tokens,device=key.device),(0,pad)).reshape(count_blocks,block_tokens)
    lengths = valid.sum(-1).long()
    ordinal = torch.arange(block_tokens,device=key.device).view(1,-1)
    labels = (ordinal*groups//lengths[:,None]).clamp_max(groups-1)
    if grouping == 'random_groups':
        generator = torch.Generator(device=key.device).manual_seed(seed)
        scores = torch.rand(count_blocks,block_tokens,device=key.device,generator=generator).masked_fill(valid == 0,2.)
        rank = scores.argsort(-1).argsort(-1)
        labels = (rank*groups//lengths[:,None]).clamp_max(groups-1)
    def assignments(labels, mask):
        return F.one_hot(labels,groups).float()*mask[...,None]
    if grouping == 'key_kmeans':
        assignment = assignments(labels, valid)
        sizes = assignment.sum(1)
        centers = torch.einsum('nlg,nld->ngd',assignment,kb)/sizes.clamp_min(1)[...,None]
        for _ in range(iterations):
            distance = (kb.square().sum(-1,keepdim=True)+centers.square().sum(-1)[:,None,:]
                        -2*torch.bmm(kb,centers.transpose(1,2)))
            labels = distance.argmin(-1)
            assignment = assignments(labels,valid)
            sizes = assignment.sum(1)
            means = torch.einsum('nlg,nld->ngd',assignment,kb)/sizes.clamp_min(1)[...,None]
            centers = torch.where(sizes[...,None] > 0,means,centers)
    remaining = torch.ones(batch,heads,tokens,device=key.device)
    remaining.scatter_(2,selected_indices.to(key.device).long(),0.)
    remaining = F.pad(remaining.reshape(batch,heads,frames,frame_tokens),(0,pad)).reshape(count_blocks,block_tokens)
    assignment = assignments(labels,remaining*valid)
    counts = assignment.sum(1)
    def tail_mean(tensor):
        mean = torch.einsum('nlg,nld->ngd',assignment,tensor)/counts.clamp_min(1)[...,None]
        return mean.reshape(batch,heads,frames*blocks*groups,dim).permute(0,2,1,3).to(key.dtype)
    tail = PrototypeTail(tail_mean(kb),tail_mean(vb),counts.reshape(batch,heads,frames*blocks*groups),block_tokens,frame_tokens)
    return tail, dict(grouping=grouping,groups_per_block=groups,iterations=iterations if grouping=='key_kmeans' else 0,
        grouping_uses_current_query=False,grouping_uses_future_video=False,
        uint8_group_index_bytes_if_persisted=batch*heads*tokens,prototype_slots=frames*blocks*groups,
        representation_bytes=tail.bytes)
