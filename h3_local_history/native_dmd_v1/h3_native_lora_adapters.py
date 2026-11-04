"""Explicit all-attention LoRA for the native causal-H3 initialization control."""
import math
import torch
from torch import nn

class NativeLoRALinear(nn.Module):
    def __init__(self,base,rank,alpha):
        super().__init__();self.base=base.requires_grad_(False);self.rank=int(rank);self.scale=float(alpha)/rank
        self.lora_a=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.lora_b=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a,a=math.sqrt(5))
    def forward(self,x):
        base=self.base(x);dtype=base.dtype
        update=torch.nn.functional.linear(torch.nn.functional.linear(x.to(dtype),self.lora_a.to(dtype)),self.lora_b.to(dtype))*self.scale
        return base+update.to(dtype)

def install_native_lora(transformer,rank=8,alpha=16.):
    if rank<1:raise ValueError('positive adapter rank required')
    transformer.requires_grad_(False);projections=[]
    for i,block in enumerate(transformer.transformer_blocks):
        attn=block.attn
        for name in ('to_q','to_k','to_v'):
            original=getattr(attn,name)
            if not isinstance(original,nn.Linear):raise ValueError('adapter already installed or unsupported projection')
            setattr(attn,name,NativeLoRALinear(original,rank,alpha));projections.append(f'transformer_blocks.{i}.attn.{name}')
        original=attn.to_out[0]
        if not isinstance(original,nn.Linear):raise ValueError('unsupported output projection')
        attn.to_out[0]=NativeLoRALinear(original,rank,alpha);projections.append(f'transformer_blocks.{i}.attn.to_out.0')
    return projections
