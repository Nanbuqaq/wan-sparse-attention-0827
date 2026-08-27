"""RoPE for per-head sparse tokens with preserved temporal/spatial positions."""

from __future__ import annotations

import torch


def build_sparse_positions(
    *,
    frame_ids: torch.Tensor,
    token_ids: torch.Tensor,
    current_frame_id: int,
    spatial_width: int,
    rope_policy: str,
    max_relative_age: int,
    candidate_frame_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``[B,H,K,3]`` temporal/height/width indices.

    ``recency_rank`` ranks candidate frames from oldest to newest and assigns
    compact non-negative positions. ``clipped_relative_age`` uses the actual
    age relative to the current chunk, clipped to the configured RoPE range.
    """

    if frame_ids.shape != token_ids.shape or frame_ids.ndim != 3:
        raise ValueError("frame_ids and token_ids must both be [B,H,K]")
    if spatial_width < 1:
        raise ValueError("spatial_width must be positive")

    if rope_policy == "upstream_zero":
        temporal = torch.zeros_like(frame_ids)
    elif rope_policy == "clipped_relative_age":
        temporal = (int(current_frame_id) - frame_ids).clamp(0, max_relative_age)
    elif rope_policy == "recency_rank":
        if candidate_frame_ids is None:
            unique = torch.unique(frame_ids.detach().cpu(), sorted=True)
        else:
            unique = torch.unique(candidate_frame_ids.detach().cpu(), sorted=True)
        rank_by_frame = {int(frame): rank for rank, frame in enumerate(unique.tolist())}
        temporal = torch.empty_like(frame_ids)
        for frame, rank in rank_by_frame.items():
            temporal[frame_ids == frame] = rank
    else:
        raise ValueError(f"unsupported rope policy: {rope_policy!r}")

    height = torch.div(token_ids, spatial_width, rounding_mode="floor")
    width = token_ids.remainder(spatial_width)
    return torch.stack((temporal, height, width), dim=-1)


def apply_selected_rope(
    key_unrotated: torch.Tensor,
    positions: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """Apply Wan causal RoPE to sparse per-head tokens.

    Args:
        key_unrotated: ``[B,K,H,D]``.
        positions: ``[B,H,K,3]`` containing temporal/height/width indices.
        freqs: Wan frequency table ``[max_position,D/2]`` (complex).
    """

    if key_unrotated.ndim != 4:
        raise ValueError("key_unrotated must be [B,K,H,D]")
    batch, tokens, heads, dim = key_unrotated.shape
    if positions.shape != (batch, heads, tokens, 3):
        raise ValueError(
            f"positions shape {tuple(positions.shape)} does not match "
            f"{(batch, heads, tokens, 3)}"
        )
    if dim % 2:
        raise ValueError("RoPE head dimension must be even")
    if not torch.is_complex(freqs):
        raise TypeError("Wan RoPE frequency table must be complex")

    complex_dim = dim // 2
    split = [complex_dim - 2 * (complex_dim // 3), complex_dim // 3, complex_dim // 3]
    temporal_freqs, height_freqs, width_freqs = freqs.split(split, dim=1)
    pos = positions.to(device=freqs.device, dtype=torch.long)
    max_indices = pos.amax(dim=(0, 1, 2))
    if int(max_indices[0]) >= temporal_freqs.shape[0]:
        raise IndexError("temporal RoPE index exceeds frequency table")
    if int(max_indices[1]) >= height_freqs.shape[0]:
        raise IndexError("height RoPE index exceeds frequency table")
    if int(max_indices[2]) >= width_freqs.shape[0]:
        raise IndexError("width RoPE index exceeds frequency table")

    rotation = torch.cat(
        (
            temporal_freqs[pos[..., 0]],
            height_freqs[pos[..., 1]],
            width_freqs[pos[..., 2]],
        ),
        dim=-1,
    ).permute(0, 2, 1, 3)
    original_dtype = key_unrotated.dtype
    key_complex = torch.view_as_complex(
        key_unrotated.to(torch.float64).reshape(batch, tokens, heads, -1, 2)
    )
    rotated = torch.view_as_real(key_complex * rotation).flatten(3)
    return rotated.to(original_dtype)

