"""Index-time RoPE-aligned prototypes for the fixed upstream-zero policy.

This is a capture-stage hypothesis, not an integrated online method. Under this
policy every historical temporal position is zero and spatial coordinates are
immutable: the rotated key mean can be constructed once when archiving a frame.
The original (unrotated) archive KV is retained for actual attention execution.
"""
from __future__ import annotations
import torch
from .rope import apply_selected_rope


def canonical_wan_frequency_table(head_dim: int, max_positions: int=1024):
    """Formula from pinned CausalWanModel initialization and rope_params."""
    if head_dim<6 or head_dim%2:
        raise ValueError('Wan multidimensional RoPE requires an even head dimension >=6')
    sections=(head_dim-4*(head_dim//6),2*(head_dim//6),2*(head_dim//6))
    parts=[]
    for dim in sections:
        angles=torch.outer(torch.arange(max_positions),1./torch.pow(10000,
                            torch.arange(0,dim,2).to(torch.float64).div(dim)))
        parts.append(torch.polar(torch.ones_like(angles),angles))
    return torch.cat(parts,dim=1)


def archive_rope0_key(key, *, spatial_height, spatial_width, freqs, rope_policy='upstream_zero'):
    """Index-time transformation only: no current Q, teacher, or future input."""
    if rope_policy!='upstream_zero':
        raise ValueError('fixed index-time rotation is only valid for upstream_zero')
    batch,tokens,heads,_=key.shape
    if tokens!=spatial_height*spatial_width:
        raise ValueError('exactly one complete archive frame is required')
    token=torch.arange(tokens,device=key.device)
    positions=torch.stack((torch.zeros_like(token),token//spatial_width,token%spatial_width),-1)
    positions=positions.view(1,1,tokens,3).expand(batch,heads,-1,-1)
    return apply_selected_rope(key,positions,freqs)


def archive_rope0_prototypes(key, *, spatial_height, spatial_width, freqs, block_tokens=64):
    rotated=archive_rope0_key(key,spatial_height=spatial_height,spatial_width=spatial_width,freqs=freqs)
    return torch.stack([rotated[:,start:start+block_tokens].float().mean(1)
                        for start in range(0,key.shape[1],block_tokens)],dim=2)
