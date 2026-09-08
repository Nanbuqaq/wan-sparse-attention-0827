from copy import deepcopy

import pytest

from scripts.collect_native_cut_screen import validate


def test_native_cut_screen_needs_observed_pinning_not_only_a_prefix():
    d=dict(status='pass',gate=False,cut_scenario='toy',seed=1,latent_shape=[1,128,48,44,80],
        local_frames=32,sink_frames=8,pixels={'frames':509},attention_backend='native_FA2',fallback_allowed=False,
        expected_scene_cut_block_indices=[3,6,12],native_shot_pin_events=[{'completed_latent':i} for i in (32,56,104)],
        upstream_source_SHA='6b36d20ec6f7958d29d11a704dfa64611a9f2572',
        strict_generator_load={'missing_keys':[],'unexpected_keys':[]},gpu='test',runner_commit='a'*40)
    assert validate(d,'toy',1)['Dense_only_no_new_memory_method']
    bad=deepcopy(d);bad['native_shot_pin_events'].pop()
    with pytest.raises(ValueError):validate(bad,'toy',1)
