from pathlib import Path

import pytest
import torch

from scripts.run_longlive2_native_reference import native_schedule, CachedNativeTextEncoder


def test_native_schedule_preserves_block_alignment_and_declares_resampling():
    root=Path(__file__).resolve().parents[1]
    segments,prompts=native_schedule(root,128)
    assert [s['start_latent'] for s in segments]==[0,24,48,80]
    assert len(prompts[0])==16 and prompts[0][10]==segments[-1]['prompt']
    _,control=native_schedule(root,128,'empty')
    assert prompts[0][:10]==control[0][:10] and prompts[0][10]!=control[0][10]


def test_native_cached_encoder_only_reuses_given_per_prompt_embeddings():
    values={'a':torch.ones(1,2,3),'b':torch.zeros(1,2,3)}
    encoder=CachedNativeTextEncoder(values,'cpu')
    out=encoder(text_prompts=['a','b','a'])['prompt_embeds']
    assert torch.equal(out,torch.cat([values[x] for x in ('a','b','a')]))
    with pytest.raises(ValueError):encoder(text_prompts=['unknown'])
