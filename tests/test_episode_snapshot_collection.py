from copy import deepcopy

import pytest

from scripts.collect_episode_snapshot import validate


def test_version_comparison_requires_same_prefix_bytes_and_slots():
    rows=[]
    for name in ('official_recache','snapshot_pre_recache','snapshot_post_recache'):
        enabled=name!='official_recache'
        rows.append(dict(variant=name,status='pass',recache_events=[{}, {}, {}],identity={'noise':'n'},
            pre_return_latent_sha256='p',history_H2D_bytes=100 if enabled else 0,archive_bytes=100 if enabled else 0,
            snapshot_capture=dict(global_frames=list(range(42,48)),budget_bytes=200,D2H_bytes=100),
            snapshot_restore=dict(current_start=78,attention_size_unchanged=True,local_slots_before_next_roll=[4,10])))
    report=dict(status='pass',seed=1,latent_frames=120,variants=rows,gpu='test',source_commit='a'*40)
    assert validate(report,1)['cases']==3
    for field,value in [('pre_return_latent_sha256','other'),('history_H2D_bytes',90)]:
        broken=deepcopy(report);broken['variants'][2][field]=value
        with pytest.raises(ValueError):validate(broken,1)
