from pathlib import Path

import pytest
import torch

from scripts.run_longlive2_native_reference import native_schedule, native_cut_schedule, CachedNativeTextEncoder


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


def test_native_cut_prefix_is_only_on_transition_blocks():
    root=Path(__file__).resolve().parents[1]
    for gate,indices in ((False,[3,6,12]),(True,[1,2,4])):
        segments,prompts=native_cut_schedule(root,'generated_patchwork_toy_cut_revisit',gate=gate)
        assert [i for i,s in enumerate(prompts[0]) if s.startswith('The scene transitions. ')]==indices
        assert len(prompts[0])==(6 if gate else 16)
        if not gate:assert segments[-1]['start_latent']-segments[2]['start_latent']==48


def test_episode_gate_has_nonresident_away_control_and_later_return():
    root=Path(__file__).resolve().parents[1]
    segments,prompts=native_cut_schedule(root,'generated_patchwork_toy_cut_revisit',gate=True,episode_gate=True)
    assert [s['start_latent'] for s in segments]==[0,8,16,48]
    assert len(prompts[0])==8
