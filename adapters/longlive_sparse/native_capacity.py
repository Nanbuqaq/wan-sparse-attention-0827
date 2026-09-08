"""Common CFG1 allocation optimization; never changes attended positive KV.

Native LongLive2 allocates both positive and negative caches even when CFG=1
never executes an unconditional forward. Keep the exact positive-cache schema,
but do not allocate the unused negative self-Attention tensors. This is a
baseline optimization, not a new admission algorithm or compression method.
"""
from types import MethodType

import torch


def initialize_positive_only(pipe, batch_size, dtype, device):
    if pipe.guidance_scale != 1.0 or pipe.quantize_kv or dtype != torch.bfloat16:
        raise ValueError('positive-only allocation requires BF16 CFG1 without KV quantization')
    if pipe.local_attn_size <= 0 or pipe.local_attn_size % pipe.num_frame_per_block:
        raise ValueError('positive-only allocation requires an explicit block-aligned window')
    heads=pipe._dit_model.num_heads;dim=pipe._dit_model.dim//heads
    tokens=pipe.local_attn_size*pipe.frame_seq_length
    block_tokens=pipe.num_frame_per_block*pipe.frame_seq_length
    positive=[]
    for _ in range(pipe.num_transformer_blocks):
        positive.append(dict(
            k=torch.zeros([batch_size,tokens,heads,dim],dtype=dtype,device=device),
            v=torch.zeros([batch_size,tokens,heads,dim],dtype=dtype,device=device),
            quantized=False,block_token_size=block_tokens,max_blocks=tokens//block_tokens,
            num_heads=heads,num_filled_blocks=0,
            global_end_index=torch.tensor([0],dtype=torch.long,device=device),
            local_end_index=torch.tensor([0],dtype=torch.long,device=device),
            pinned_start=torch.tensor([-1],dtype=torch.long,device=device),
            pinned_len=torch.tensor([0],dtype=torch.long,device=device)))
    pipe.kv_cache_pos=positive
    pipe.kv_cache_neg=[]


def install_positive_only_allocator(pipe):
    if pipe.guidance_scale != 1.0 or pipe.quantize_kv:
        raise ValueError('cannot discard a required unconditional or quantized cache')
    if pipe.kv_cache_pos is not None or pipe.kv_cache_neg is not None:
        raise ValueError('install allocation policy before generation, never discard live state')
    pipe._initialize_kv_cache=MethodType(initialize_positive_only,pipe)
