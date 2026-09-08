"""Compact committed prototype wire format and whole-Block64 precision.

Whole-block raw admission removes all of that block's prototypes, avoiding
subtraction of nearly equal rounded sums. Non-admitted blocks remain as four
BF16 K/V means with count weights. The third BF16 vector is K variance for
routing, not another attention key. This is not an exact Dense operator.
"""
from dataclasses import dataclass
import torch

from .committed_moments import build_frame_moments


@dataclass
class PrototypeWireFrame:
    key_mean: torch.Tensor       # [B,H,P,D], BF16
    value_mean: torch.Tensor
    key_variance: torch.Tensor
    counts: torch.Tensor         # [B,H,P], int16
    frame_tokens: int
    block_tokens: int
    groups: int
    variance_scale: torch.Tensor | None = None

    @property
    def bytes(self):
        tensors=[self.key_mean,self.value_mean,self.key_variance,self.counts]
        if self.variance_scale is not None:tensors.append(self.variance_scale)
        return sum(t.numel()*t.element_size() for t in tensors)

    def decoded_variance(self):
        values=self.key_variance.float()
        return values if self.variance_scale is None else values*self.variance_scale

    def cpu(self):
        return PrototypeWireFrame(*(t.cpu() for t in (self.key_mean,self.value_mean,self.key_variance,self.counts)),
            self.frame_tokens,self.block_tokens,self.groups,
            self.variance_scale.cpu() if self.variance_scale is not None else None)


def encode_prototype_frame(key,value,*,block_tokens=64,groups=4,variance_codec='bf16'):
    if variance_codec not in ('bf16','u8_scaled'):raise ValueError('unknown variance wire codec')
    moments=build_frame_moments(key,value,block_tokens=block_tokens,grouping='key_kmeans',groups=groups)
    divisor=moments.counts.clamp_min(1)[...,None]
    mean=moments.key_sum/divisor
    labels=moments.token_groups.long()[...,None].expand(-1,-1,-1,key.shape[-1])
    second=torch.zeros_like(moments.key_sum).scatter_add_(2,labels,key.permute(0,2,1,3).float().square())/divisor
    variance=(second-mean.square()).clamp_min(0)
    scale=None
    if variance_codec=='u8_scaled':
        scale=variance.amax(-1,keepdim=True).div(255).clamp_min(1e-12)
        encoded=(variance/scale).round().clamp(0,255).to(torch.uint8)
    else:encoded=variance.to(torch.bfloat16)
    return PrototypeWireFrame(mean.to(torch.bfloat16),(moments.value_sum/divisor).to(torch.bfloat16),
        encoded,moments.counts.to(torch.int16),key.shape[1],block_tokens,groups,scale)


def choose_whole_blocks(scores,widths,*,token_budget,byte_normalized=True):
    """Select no more than the physical raw budget; never split a block.

    This uses known raw KV byte cost only, not the rejected exposed-wait model.
    Frame-tail blocks may have fewer than64 tokens. Return nested block lists
    and per-head actual counts; padding is the caller's explicit responsibility.
    """
    if scores.ndim!=3 or scores.device.type!='cpu' or widths.ndim!=1 or widths.numel()!=scores.shape[-1]:
        raise ValueError('CPU scores [B,H,blocks] and matching block widths required')
    if token_budget<0 or bool((widths<=0).any()) or not torch.isfinite(scores).all():
        raise ValueError('invalid raw budget, widths, or scores')
    ranking=scores/widths if byte_normalized else scores
    orders=torch.argsort(ranking,dim=-1,descending=True,stable=True).tolist()
    sizes=widths.tolist()
    selected=[];counts=torch.zeros(scores.shape[:2],dtype=torch.long)
    for b in range(scores.shape[0]):
        selected_heads=[]
        for h in range(scores.shape[1]):
            left=int(token_budget);blocks=[]
            for block in orders[b][h]:
                size=sizes[block]
                if size<=left:
                    blocks.append(block);left-=size
                if left==0:break
            selected_heads.append(sorted(blocks));counts[b,h]=token_budget-left
        selected.append(selected_heads)
    return selected,counts


def whole_block_token_indices(blocks,*,frame_tokens,frames,block_tokens=64):
    per_frame=(frame_tokens+block_tokens-1)//block_tokens
    result=[]
    for block in blocks:
        if block<0 or block>=frames*per_frame:raise ValueError('block outside candidate set')
        frame,within=divmod(block,per_frame)
        start=within*block_tokens;length=min(block_tokens,frame_tokens-start)
        result.append(torch.arange(frame*frame_tokens+start,frame*frame_tokens+start+length))
    if not result:return torch.empty(0,dtype=torch.long)
    return torch.cat(result).sort().values
