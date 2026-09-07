import pytest
import torch

from adapters.longlive_sparse.streaming_vae import StreamingVAEDecoder


class DummyCache:
    def __init__(self):
        self.seen = 0

    def clear_cache(self):
        self.seen = 0

    def cached_decode(self, chunk, scale):
        del scale
        length = chunk.shape[2]
        pixels = 4*length-(3 if self.seen == 0 else 0)
        self.seen += length
        return torch.zeros(1, 3, pixels, 2, 2)


class DummyVAE:
    def __init__(self):
        self.mean, self.std = torch.zeros(1), torch.ones(1)
        self.model = DummyCache()


def test_streaming_vae_preserves_continuity_and_callback_order():
    seen = []
    vae = DummyVAE()
    decoder = StreamingVAEDecoder(vae, device='cpu', on_chunk=lambda pixels, row: seen.append(row['start_pixel']))
    for start in (0, 3, 6):
        decoder.submit(torch.zeros(1, 3, 1, 2, 2), start_latent=start)
    output, metrics = decoder.finish()
    assert output.shape == (1, 33, 3, 2, 2)
    assert seen == [0, 9, 21]
    assert metrics['completed_latents'] == 9 and metrics['completed_pixels'] == 33
    assert metrics['pinned_output_bytes'] == 0


def test_streaming_sink_can_avoid_collecting_outputs():
    seen = []
    decoder = StreamingVAEDecoder(DummyVAE(), device='cpu', collect=False, on_chunk=lambda x, row: seen.append(x.shape[1]))
    decoder.submit(torch.zeros(1, 3, 1, 2, 2), start_latent=0)
    output, metrics = decoder.finish()
    assert output is None and seen == [9]
    assert not metrics['collected_CPU_outputs_unbounded_by_slot_budget']


def test_out_of_order_completed_chunks_are_rejected():
    decoder = StreamingVAEDecoder(DummyVAE(), device='cpu')
    with pytest.raises(ValueError, match='in order'):
        decoder.submit(torch.zeros(1, 3, 1, 2, 2), start_latent=3)
    decoder.submit(torch.zeros(1, 3, 1, 2, 2), start_latent=0)
    decoder.finish()
