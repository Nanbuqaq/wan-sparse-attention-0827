"""Count-weighted history prototypes after a raw route has been chosen.

Full-candidate construction is materialization/reference work, NOT an online
selector. Charge its complete preparation and transfer costs.
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class PrototypeTail:
    key: torch.Tensor
    value: torch.Tensor
    counts: torch.Tensor
    block_tokens: int
    frame_tokens: int

    @property
    def bytes(self):
        return sum(x.numel()*x.element_size() for x in (self.key, self.value, self.counts))


def build_prototype_tail(key, value, selected_indices, *, frame_tokens, block_tokens=64,
                         prototype_dtype=torch.bfloat16):
    if key.ndim != 4 or value.shape != key.shape or key.shape[1] % frame_tokens:
        raise ValueError('complete frame-major K/V candidates required')
    batch, tokens, heads, dim = key.shape
    if selected_indices.shape[:2] != (batch, heads):
        raise ValueError('selected index batch/head mismatch')
    selected = selected_indices.to(key.device).long()
    if selected.numel() and (int(selected.min()) < 0 or int(selected.max()) >= tokens):
        raise ValueError('invalid selected raw coordinate')
    if selected.shape[-1] > 1 and bool((selected.sort(-1).values.diff(dim=-1) == 0).any()):
        raise ValueError('duplicate raw coordinate would double-count mass')
    keep = torch.ones(batch, heads, tokens, device=key.device, dtype=torch.float32)
    keep.scatter_(2, selected, 0.)
    frames = tokens//frame_tokens
    blocks = (frame_tokens+block_tokens-1)//block_tokens
    pad = blocks*block_tokens-frame_tokens
    mask = keep.reshape(batch, heads, frames, frame_tokens)
    counts = F.pad(mask, (0, pad)).reshape(batch, heads, frames, blocks, block_tokens).sum(-1)
    def means(tensor):
        matrix = tensor.permute(0, 2, 1, 3).float().reshape(batch, heads, frames, frame_tokens, dim)*mask[..., None]
        sums = F.pad(matrix, (0, 0, 0, pad)).reshape(batch, heads, frames, blocks, block_tokens, dim).sum(-2)
        result = sums/counts.clamp_min(1)[..., None]
        return result.reshape(batch, heads, frames*blocks, dim).permute(0, 2, 1, 3).to(prototype_dtype)
    return PrototypeTail(means(key), means(value), counts.reshape(batch, heads, frames*blocks), block_tokens, frame_tokens)


def execute_weighted_tail_sdpa(query, exact_key, exact_value, history_key, history_value, tail, *, efficient_only=True,
                               raw_valid_counts=None):
    """Compact per-key bias, never an explicit Q-by-K bias tensor."""
    key = torch.cat((exact_key, history_key, tail.key.to(query.dtype)), dim=1)
    value = torch.cat((exact_value, history_value, tail.value.to(query.dtype)), dim=1)
    batch, _, heads, _ = query.shape
    raw, total = exact_key.shape[1]+history_key.shape[1], key.shape[1]
    # Align head strides for memory-efficient SDPA without adding logical keys.
    bias_storage = torch.zeros(batch, heads, 1, ((total+7)//8)*8, device=query.device, dtype=query.dtype)
    bias = bias_storage[..., :total]
    if raw_valid_counts is not None:
        valid=raw_valid_counts.to(query.device)
        if valid.shape!=(batch,heads) or bool((valid<0).any()) or bool((valid>history_key.shape[1]).any()):
            raise ValueError('raw valid counts must match per-head history lengths')
        slots=torch.arange(history_key.shape[1],device=query.device)
        bias[...,exact_key.shape[1]:raw]=torch.where(slots[None,None,None,:]<valid[:,:,None,None],0.,-float('inf'))
    bias[..., raw:] = tail.counts[:, :, None].float().log()
    if query.is_cuda and efficient_only:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            output = F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2),
                value.transpose(1, 2), attn_mask=bias, dropout_p=0.)
    else:
        output = F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2),
            value.transpose(1, 2), attn_mask=bias, dropout_p=0.)
    return output.transpose(1, 2)
