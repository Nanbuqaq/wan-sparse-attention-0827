import hashlib
import json

import pytest

from scripts.run_native_visual_restart import read_spec


def write_spec(tmp_path, **overrides):
    image=tmp_path/'past.png';image.write_bytes(b'fixed past image')
    spec=dict(schema='causal_visual_restart_v1',prompt='The arrived request',image='past.png',
        image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),source_pixel_index=188,
        source_committed_latents=48,request_arrival_latent=96)
    spec.update(overrides);path=tmp_path/'spec.json';path.write_text(json.dumps(spec));return path


def test_committed_past_boundary(tmp_path):
    spec,image=read_spec(write_spec(tmp_path))
    assert spec['source_pixel_index']==188 and image.name=='past.png'


@pytest.mark.parametrize('change',[dict(source_pixel_index=189),dict(source_pixel_index=-1),
    dict(source_committed_latents=97),dict(image_sha256='changed'),
    dict(prompt='The scene transitions. Fresh request'),dict(prompt='')])
def test_rejects_future_or_changed_input(tmp_path,change):
    with pytest.raises(ValueError):read_spec(write_spec(tmp_path,**change))
