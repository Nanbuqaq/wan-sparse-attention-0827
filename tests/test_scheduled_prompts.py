from types import SimpleNamespace
import pytest
import torch
from adapters.longlive_sparse.scheduled_prompts import ScheduledPromptContext,validate_schedule


class Encoder(torch.nn.Module):
    def __init__(self):super().__init__();self.calls=[]
    def forward(self,text_prompts):
        self.calls.append(text_prompts)
        return {'prompt_embeds':text_prompts[0]}


class Generator(torch.nn.Module):
    def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1))
    def forward(self,**kwargs):return kwargs


def test_only_current_prompt_exposed_and_text_cache_invalidated_once():
    pipeline=SimpleNamespace(text_encoder=Encoder(),generator=Generator(),num_frame_per_block=3,frame_seq_length=10)
    cache=[{'is_init':True,'k':torch.ones(2)} for _ in range(2)]
    segments=[{'start_latent':0,'prompt':'first'},{'start_latent':6,'prompt':'second'},{'start_latent':12,'prompt':'third'}]
    with ScheduledPromptContext(pipeline,segments,latent_frames=18) as context:
        assert pipeline.text_encoder(text_prompts=['first'])['prompt_embeds']=='first'
        for start,expected in [(0,'first'),(30,'first'),(60,'second'),(60,'second'),(90,'second'),(120,'third')]:
            out=pipeline.generator(current_start=start,conditional_dict={},crossattn_cache=cache)
            assert out['conditional_dict']['prompt_embeds']==expected
            assert all(torch.equal(c['k'],torch.ones(2)) for c in cache)
        assert [e['crossattn_cache_resets'] for e in context.events]==[0,2,2]
    assert pipeline.text_encoder(text_prompts=['restored'])['prompt_embeds']=='restored'


def test_duplicate_text_events_are_noop_and_bad_first_prompt_rejected():
    pipeline=SimpleNamespace(text_encoder=Encoder(),generator=Generator(),num_frame_per_block=3,frame_seq_length=10)
    with ScheduledPromptContext(pipeline,[{'start_latent':0,'prompt':'same'},{'start_latent':6,'prompt':'same'}],latent_frames=12) as context:
        cache=[{'is_init':True}]
        pipeline.generator(current_start=0,conditional_dict={},crossattn_cache=cache)
        pipeline.generator(current_start=60,conditional_dict={},crossattn_cache=cache)
        assert cache[0]['is_init'] and context.encoded_unique_prompts==1
        with pytest.raises(ValueError):pipeline.text_encoder(text_prompts=['wrong'])


@pytest.mark.parametrize('starts',[[3],[0,4],[0,6,3],[0,3,3]])
def test_invalid_schedule_rejected(starts):
    with pytest.raises(ValueError):validate_schedule([{'start_latent':s,'prompt':'x'} for s in starts])
