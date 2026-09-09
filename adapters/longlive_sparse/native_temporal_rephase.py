"""Relative temporal RoPE rotation of archived keys; not bit recovery of raw K.

Matches the locked Wan5B head128 partition: 44 temporal real channels, 42+42
spatial channels. Values and spatial key channels are never transformed.
"""
import torch


def rephase_temporal_keys(key,delta,theta=10000.):
    if key.shape[-1]!=128:raise ValueError('native rephase currently requires head128')
    if float(delta)==0:return key
    temporal=44
    angle=torch.arange(0,temporal,2,device=key.device,dtype=torch.float64)/temporal
    rotation=torch.polar(torch.ones_like(angle),float(delta)/torch.pow(theta,angle))
    pairs=torch.view_as_complex(key[...,:temporal].double().contiguous().reshape(*key.shape[:-1],temporal//2,2))
    changed=torch.view_as_real(pairs*rotation).flatten(-2).to(key.dtype)
    result=key.clone();result[...,:temporal]=changed
    return result
