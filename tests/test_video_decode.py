from __future__ import annotations

import torch

from adapters.longlive_sparse.video_decode import (
    decode_latents_chunked_exact,
    expected_pixel_frames,
)


class DummyCachedDecoder:
    def __init__(self):
        self.seen = 0
        self.clear_calls = 0

    def clear_cache(self):
        self.seen = 0
        self.clear_calls += 1

    def cached_decode(self, chunk, scale):
        del scale
        latent_frames = int(chunk.shape[2])
        pixel_frames = 4 * latent_frames - (3 if self.seen == 0 else 0)
        self.seen += latent_frames
        return torch.zeros((1, 3, pixel_frames, 2, 2))


class DummyVAE:
    def __init__(self):
        self.mean = torch.zeros(1)
        self.std = torch.ones(1)
        self.model = DummyCachedDecoder()


def test_cache_continuous_chunk_decode_preserves_957_frames():
    vae = DummyVAE()
    latent = torch.zeros((1, 240, 16, 2, 2), dtype=torch.bfloat16)
    video = decode_latents_chunked_exact(vae, latent, chunk_size=120)
    assert video.shape == (1, 957, 3, 2, 2)
    assert expected_pixel_frames(240) == 957
    assert vae.model.clear_calls >= 2
