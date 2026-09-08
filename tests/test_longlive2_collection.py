from copy import deepcopy

import pytest

from scripts.collect_longlive2_reference import validate


def test_native_reference_requires_full_shape_strict_load_and_no_fallback():
    d=dict(status='pass',seed=1,control=None,gate=False,latent_shape=[1,128,48,44,80],
        local_frames=32,sink_frames=8,pixel_frames=509,pixels={'frames':509},
        upstream_source_SHA='6b36d20ec6f7958d29d11a704dfa64611a9f2572',attention_backend='native_FA2',
        fallback_allowed=False,strict_generator_load={'missing_keys':[],'unexpected_keys':[]},gpu='test',runner_commit='a'*40)
    assert validate(d,1,None)['pixel_frames']==509
    for key,value in [('local_frames',16),('fallback_allowed',True),('pixel_frames',511)]:
        bad=deepcopy(d);bad[key]=value
        with pytest.raises(ValueError):validate(bad,1,None)
