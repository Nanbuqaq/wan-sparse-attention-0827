"""Additive compressed history built when a frame is committed.

The original KV stays in the existing CPU archive. Moment construction sees
that completed frame once; tail reconstruction reads only its small moments,
coordinate labels, and already-admitted raw KV. No full-candidate onload.
RoPE must be applied before building moments; temporal-policy validity belongs
to the runtime owner. These are exact first moments, not exact attention.
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from .prototype_tail import PrototypeTail


@dataclass
class FrameMoments:
    key_sum: torch.Tensor       # [B,H,P,D], FP32
    value_sum: torch.Tensor
    counts: torch.Tensor       # [B,H,P], FP32
    token_groups: torch.Tensor # [B,H,T], integer local prototype IDs
    frame_tokens: int
    block_tokens: int
    grouping: str

    @property
    def bytes(self):
        return sum(t.numel()*t.element_size() for t in
                   (self.key_sum, self.value_sum, self.counts, self.token_groups))

    def cpu(self):
        return FrameMoments(*(t.cpu() for t in
            (self.key_sum, self.value_sum, self.counts, self.token_groups)),
            self.frame_tokens, self.block_tokens, self.grouping)


def build_frame_moments(key, value, *, block_tokens=64, grouping='spatial',
                        groups=4, iterations=4):
    """One complete, already-rotated frame; grouping uses K only, never Q."""
    if key.ndim != 4 or value.shape != key.shape or key.shape[1] < 1:
        raise ValueError('one nonempty frame of matching [B,T,H,D] KV required')
    if grouping not in ('spatial', 'key_kmeans') or block_tokens < 1 or groups < 1 or iterations < 0:
        raise ValueError('invalid grouping configuration')
    batch, tokens, heads, dim = key.shape
    blocks = (tokens+block_tokens-1)//block_tokens
    pad = blocks*block_tokens-tokens
    slots_per_block = 1 if grouping == 'spatial' else groups
    def view(t):
        return F.pad(t.permute(0,2,1,3).float(), (0,0,0,pad)).reshape(-1,block_tokens,dim)
    keys, values = view(key), view(value)
    valid = F.pad(torch.ones(batch,heads,tokens,device=key.device),(0,pad)).reshape(-1,block_tokens)
    lengths = valid.sum(-1).long()
    ordinal = torch.arange(block_tokens,device=key.device)[None]
    labels = (ordinal*slots_per_block//lengths[:,None]).clamp_max(slots_per_block-1)
    def assignment(labels):
        return F.one_hot(labels,slots_per_block).float()*valid[...,None]
    if grouping == 'key_kmeans':
        weights = assignment(labels)
        counts = weights.sum(1)
        centers = torch.einsum('ntp,ntd->npd',weights,keys)/counts.clamp_min(1)[...,None]
        for _ in range(iterations):
            distances = keys.square().sum(-1,keepdim=True)+centers.square().sum(-1)[:,None]-2*torch.bmm(keys,centers.transpose(1,2))
            labels = distances.argmin(-1)
            weights = assignment(labels)
            counts = weights.sum(1)
            means = torch.einsum('ntp,ntd->npd',weights,keys)/counts.clamp_min(1)[...,None]
            centers = torch.where(counts[...,None] > 0, means, centers)
    weights = assignment(labels)
    counts = weights.sum(1).reshape(batch,heads,blocks*slots_per_block)
    def reduce(t):
        return torch.einsum('ntp,ntd->npd',weights,t).reshape(batch,heads,blocks*slots_per_block,dim)
    groups_by_token = labels.reshape(batch,heads,blocks,block_tokens)
    groups_by_token = groups_by_token + torch.arange(blocks,device=key.device)[None,None,:,None]*slots_per_block
    groups_by_token = groups_by_token.reshape(batch,heads,-1)[...,:tokens]
    label_dtype = torch.int16 if blocks*slots_per_block <= 32767 else torch.int32
    return FrameMoments(reduce(keys),reduce(values),counts,groups_by_token.to(label_dtype),
                        tokens,block_tokens,grouping)


def gather_moment_payload(frames, frame_ids, selected_frame_ids, selected_token_ids):
    """CPU metadata/summary gather; never touches original unselected KV.

    Returns four CPU tensors. Charge their packing, allocation, transfer, and
    GPU subtraction, not only the eventual attention kernel.
    """
    if not frames or len(frames) != len(frame_ids) or len(set(frame_ids)) != len(frame_ids):
        raise ValueError('distinct frame IDs and matching moment objects required')
    if selected_frame_ids.shape != selected_token_ids.shape or selected_frame_ids.ndim != 3:
        raise ValueError('selected coordinates must share [B,H,U] shape')
    if selected_frame_ids.device.type != 'cpu' or selected_token_ids.device.type != 'cpu':
        raise ValueError('coordinate metadata must be on CPU')
    first = frames[0]
    for frame in frames:
        if (frame.key_sum.device.type != 'cpu' or frame.key_sum.shape != first.key_sum.shape
                or frame.frame_tokens != first.frame_tokens or frame.grouping != first.grouping
                or frame.block_tokens != first.block_tokens):
            raise ValueError('homogeneous CPU frame moments required')
    if selected_frame_ids.shape[:2] != first.counts.shape[:2]:
        raise ValueError('coordinate batch/head mismatch')
    tokens = first.frame_tokens
    if selected_token_ids.numel() and (int(selected_token_ids.min()) < 0 or int(selected_token_ids.max()) >= tokens):
        raise ValueError('token outside frame')
    ids = torch.tensor(frame_ids, dtype=torch.long)
    sorted_ids, order = ids.sort()
    lookup = torch.searchsorted(sorted_ids, selected_frame_ids.contiguous())
    if bool((lookup >= len(frames)).any()) or not torch.equal(sorted_ids[lookup], selected_frame_ids):
        raise ValueError('selected frame missing from moments')
    frame_ordinals = order[lookup]
    flat_coordinates = frame_ordinals*tokens+selected_token_ids
    if flat_coordinates.shape[-1] > 1 and bool((flat_coordinates.sort(-1).values.diff(dim=-1) == 0).any()):
        raise ValueError('duplicate admitted token')
    slots = first.counts.shape[-1]
    labels = torch.cat([frame.token_groups.long()+i*slots for i,frame in enumerate(frames)], -1)
    selected_groups = labels.gather(2, flat_coordinates)
    return (torch.cat([f.key_sum for f in frames],2), torch.cat([f.value_sum for f in frames],2),
            torch.cat([f.counts for f in frames],2), selected_groups.to(torch.int32))


def subtract_admitted_raw(payload, raw_key, raw_value, *, frame_tokens, block_tokens,
                          prototype_dtype=torch.bfloat16):
    """Subtract admitted raw values so they are not counted again in a tail."""
    keys, values, full_counts, selected_groups = payload
    if raw_key.ndim != 4 or raw_value.shape != raw_key.shape:
        raise ValueError('raw KV shape mismatch')
    batch, admitted, heads, dim = raw_key.shape
    if (selected_groups.shape != (batch,heads,admitted) or keys.shape != values.shape
            or keys.shape[:2] != (batch,heads) or keys.shape[-1] != dim or keys.shape[:-1] != full_counts.shape):
        raise ValueError('payload/selection geometry mismatch')
    if any(t.device != raw_key.device for t in payload):
        raise ValueError('caller must explicitly transfer and account for payload')
    groups = selected_groups.long()
    if groups.numel() and (int(groups.min()) < 0 or int(groups.max()) >= full_counts.shape[-1]):
        raise ValueError('invalid prototype index')
    counts = full_counts.clone()
    counts.scatter_add_(2, groups, -torch.ones_like(groups,dtype=torch.float32))
    if bool((counts < 0).any()):
        raise ValueError('admitted raw tokens exceed prototype multiplicity')
    def tail_mean(total, raw):
        sums = total.clone()
        sums.scatter_add_(2, groups[...,None].expand(-1,-1,-1,dim), -raw.permute(0,2,1,3).float())
        means = torch.where(counts[...,None] > 0, sums/counts.clamp_min(1)[...,None], 0.)
        return means.permute(0,2,1,3).to(prototype_dtype)
    return PrototypeTail(tail_mean(keys,raw_key),tail_mean(values,raw_value),counts,block_tokens,frame_tokens)
