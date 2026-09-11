"""Fixed coordinate priorities for background fill independent of mask size."""
import torch


def stable_mask_indices(foreground, *, source_tokens, budget):
    ids = torch.as_tensor(foreground, dtype=torch.long, device='cpu').flatten()
    if (not ids.numel() or ids.unique().numel() != ids.numel()
        or int(ids.min()) < 0 or int(ids.max()) >= source_tokens):
        raise ValueError('unique nonempty original foreground coordinates required')
    if not ids.numel() <= budget <= source_tokens:
        raise ValueError('foreground does not fit the exact source budget')
    # Bit reversal distributes early priorities over the whole fixed source axis.
    # The priority does not depend on mask membership or remaining fill count.
    width = max(1, (source_tokens - 1).bit_length())
    priority = torch.arange(1 << width, dtype=torch.long)
    reversed_bits = torch.zeros_like(priority)
    for bit in range(width):
        reversed_bits |= ((priority >> bit) & 1) << (width - 1 - bit)
    priority = reversed_bits[reversed_bits < source_tokens]
    occupied = torch.zeros(source_tokens, dtype=torch.bool)
    occupied[ids] = True
    fill = priority[~occupied[priority]][:budget - ids.numel()]
    return torch.cat((ids, fill)).sort().values
