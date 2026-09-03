"""Memory-bounded VAE decoding with continuous temporal cache semantics."""

from __future__ import annotations

import torch


def expected_pixel_frames(latent_frames: int) -> int:
    if int(latent_frames) <= 0:
        raise ValueError("latent_frames must be positive")
    return 4 * int(latent_frames) - 3


def decode_latents_chunked_exact(
    vae,
    latent: torch.Tensor,
    *,
    chunk_size: int = 120,
) -> torch.Tensor:
    """Decode long latents without losing three frames at chunk boundaries.

    Wan's first latent frame decodes to one pixel frame and every subsequent
    latent frame decodes to four.  Clearing the temporal cache for every chunk
    incorrectly repeats the one-frame initialization.  Keep ``cached_decode``
    state continuous across chunks so T latents always yield ``4*T-3`` frames.
    """

    if latent.ndim != 5:
        raise ValueError(f"expected [B,T,C,H,W] latent, got {tuple(latent.shape)}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    zs = latent.permute(0, 2, 1, 3, 4)
    device, dtype = latent.device, latent.dtype
    scale = [
        vae.mean.to(device=device, dtype=dtype),
        1.0 / vae.std.to(device=device, dtype=dtype),
    ]
    outputs = []
    vae.model.clear_cache()
    try:
        for sample in zs:
            decoded_chunks = []
            for start in range(0, sample.shape[1], chunk_size):
                chunk = sample[:, start : start + chunk_size].unsqueeze(0)
                decoded = (
                    vae.model.cached_decode(chunk, scale)
                    .float()
                    .clamp_(-1, 1)
                    .squeeze(0)
                    .cpu()
                )
                decoded_chunks.append(decoded)
            outputs.append(torch.cat(decoded_chunks, dim=1))
            vae.model.clear_cache()
    finally:
        vae.model.clear_cache()
    output = torch.stack(outputs, dim=0).permute(0, 2, 1, 3, 4)
    observed = int(output.shape[1])
    expected = expected_pixel_frames(int(latent.shape[1]))
    if observed != expected:
        raise RuntimeError(
            f"cache-continuous VAE decode produced {observed} frames, expected {expected}"
        )
    return output
