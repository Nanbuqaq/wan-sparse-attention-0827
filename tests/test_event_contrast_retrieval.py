from types import SimpleNamespace
import torch
from adapters.longlive_sparse.event_contrast_retrieval import EventContrastRetrieval


class Generator(torch.nn.Module):
    def forward(self,**kwargs):return kwargs


def test_selector_uses_past_episodes_without_workload_roles():
    pipeline=SimpleNamespace(generator=Generator(),frame_seq_length=10,num_frame_per_block=3,
        sparse_history_archive=SimpleNamespace(frame_ids=lambda layer:list(range(1,30))),
        sparse_history_modules=[SimpleNamespace(layer_id=0,sink_size=1,memory_size=2)])
    c0={'prompt_embeds':torch.tensor([[[1.,0.]]])}
    c1={'prompt_embeds':torch.tensor([[[0.,1.]]])}
    c2={'prompt_embeds':torch.tensor([[[1.,0.]]])}
    original=torch.tensor([[20,21]])
    with EventContrastRetrieval(pipeline) as selector:
        for frame,condition in ((0,c0),(6,c1)):
            result=pipeline.generator(current_start=frame*10,conditional_dict=condition,memory_indices=original,
                noisy_image_or_video=torch.zeros(1,3,1))
            assert torch.equal(result['memory_indices'],original)
        current=pipeline.generator(current_start=120,conditional_dict=c2,memory_indices=original,noisy_image_or_video=torch.zeros(1,3,1))
        assert torch.equal(current['memory_indices'],torch.tensor([[3,4]])) # past frames4,5
        later=pipeline.generator(current_start=150,conditional_dict=c2,memory_indices=original,noisy_image_or_video=torch.zeros(1,3,1))
        assert torch.equal(later['memory_indices'],original)
        assert selector.audit()['events'][-1]['chosen_episode']==0
        assert selector.audit()['workload_role_labels_used'] is False
