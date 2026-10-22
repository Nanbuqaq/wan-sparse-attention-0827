"""Explicit optional attention kernels; torch Flash remains the default."""
import torch


def flash_attention(q, k, v, backend='torch_flash'):
    """Inputs/outputs BHSD, unmasked attention with the caller's exact KV span."""
    if backend == 'flash2_deterministic':
        if not q.is_cuda:
            raise ValueError('Explicit deterministic FlashAttention 2 requires CUDA')
        from flash_attn import flash_attn_func
        return flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                               dropout_p=0., causal=False, deterministic=True).transpose(1, 2)
    if backend == 'flash3':
        if not q.is_cuda:
            raise ValueError('FlashAttention 3 requires CUDA')
        from flash_attn_interface import flash_attn_func
        return flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                               causal=False, deterministic=torch.are_deterministic_algorithms_enabled()).transpose(1, 2)
    if backend != 'torch_flash':
        raise ValueError(f'Unknown attention backend: {backend}')
    if q.is_cuda:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.)
