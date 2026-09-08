from types import SimpleNamespace
import pytest
import torch
from adapters.longlive_sparse.episode_anchor_probe import EpisodeAnchorProbe


class Generator(torch.nn.Module):
    def forward(self,**kwargs):return kwargs


def test_only_declared_interval_changes_past_frame_indices():
    archive=SimpleNamespace(frame_ids=lambda layer:list(range(1,9)))
    pipeline=SimpleNamespace(generator=Generator(),frame_seq_length=10,sparse_history_archive=archive,
        sparse_history_modules=[SimpleNamespace(layer_id=0,sink_size=1,memory_size=2)])
    values=dict(noisy_image_or_video=torch.zeros(1,3,1),memory_indices=torch.tensor([[6,7]]))
    with EpisodeAnchorProbe(pipeline,anchor_frames=[3,4],start_latent=9,end_latent=12) as probe:
        before=pipeline.generator(current_start=60,**values)
        during=pipeline.generator(current_start=90,**values)
        after=pipeline.generator(current_start=120,**values)
        assert torch.equal(before['memory_indices'],values['memory_indices'])
        assert torch.equal(during['memory_indices'],torch.tensor([[2,3]]))
        assert torch.equal(after['memory_indices'],values['memory_indices'])
        assert probe.audit()['privileged_episode_interval']


def test_future_anchor_rejected():
    with pytest.raises(ValueError):EpisodeAnchorProbe(None,anchor_frames=[10],start_latent=9,end_latent=12)
