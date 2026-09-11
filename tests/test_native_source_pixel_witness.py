import torch
import pytest
from adapters.longlive_sparse.native_source_pixel_witness import NativeSourcePixelWitness


def feed(w,end):
    for i in range(end):
        x=torch.full((1,1,3),i,dtype=torch.uint8);w.on_rgb(i,x);x.zero_()


@pytest.mark.parametrize('register_first',[True,False])
def test_source_snapshot_is_owned_and_handles_either_event_order(tmp_path,register_first):
    w=NativeSourcePixelWitness(started=0,byte_budget=1000)
    w.on_latent(0,8,'a'*64);w.on_latent(8,8,'b'*64)
    if register_first:w.register(1,16,8.)
    feed(w,61)
    if not register_first:w.register(1,16,8.)
    report=w.export(tmp_path/'raw.pt',expected_archives=1)
    data=torch.load(tmp_path/'raw.pt',weights_only=True)['records'][0]
    assert data['pixels'][:,0,0,0].tolist()==list(range(29,61))
    assert data['source_latent_sha256']=='b'*64 and report['complete']
    assert len(w.ring)==32 and report['CPU_owned_peak_bytes']<=1000


def test_budget_and_missing_source_fail_explicitly(tmp_path):
    w=NativeSourcePixelWitness(started=0,byte_budget=150)
    w.on_latent(0,8,'a'*64);w.on_latent(8,8,'b'*64);w.register(1,16,8.)
    with pytest.raises(RuntimeError,match='byte budget'):feed(w,61)
    with pytest.raises(RuntimeError,match='incomplete'):w.export(tmp_path/'missing.pt',expected_archives=1)
    assert not w.export(tmp_path/'partial.pt',expected_archives=1,allow_partial=True)['complete']


def test_caps_and_stream_order_are_enforced():
    w=NativeSourcePixelWitness(started=0,max_archives=1)
    w.register(1,16,8.)
    with pytest.raises(RuntimeError,match='archive-count'):w.register(2,24,16.)
    with pytest.raises(ValueError,match='order'):w.on_rgb(2,torch.zeros(1,1,3,dtype=torch.uint8))


def test_scene_hook_uses_actual_descriptor_and_restores_original():
    from types import SimpleNamespace
    scene=SimpleNamespace(banks=[])
    def original(frame):
        scene.banks.append({'descriptor':SimpleNamespace(archive_version=7,source_end=frame,source_phase=8.)})
    scene._archive_last_scene=original
    w=NativeSourcePixelWitness(started=0);w.attach_scene(scene);scene._archive_last_scene(16)
    assert w.pending[7]['source_start']==8 and w.pending[7]['pixel_start']==29
    w.detach();assert scene._archive_last_scene is original
