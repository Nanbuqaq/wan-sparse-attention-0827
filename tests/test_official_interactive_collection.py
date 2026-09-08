import pytest
from scripts.collect_official_interactive import validate


def test_official_interactive_control_must_exercise_switches_without_archive():
    rows=[dict(variant=name,status='pass',history_H2D_bytes=0,archive_bytes=0,recache_events=[{}, {}, {}])
          for name in ('cross_only','official_recache')]
    report=dict(status='pass',seed=1,latent_frames=120,novel_control=None,variants=rows,gpu='test',
        source_commit='a'*40,upstream_interactive_sha256='b'*64)
    assert validate(report,1,None)['cases']==2
    rows[1]['archive_bytes']=1
    with pytest.raises(ValueError):validate(report,1,None)
