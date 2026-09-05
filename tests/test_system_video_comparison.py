import hashlib
import json
import torch
import pytest
from scripts.audit_system_video_comparison import compare


def case(root, *, changed_route=False, capture=False):
    root.mkdir()
    key={'commit':'a'*40}
    identity=hashlib.sha256(json.dumps(key,separators=(',',':'),sort_keys=True).encode()).hexdigest()
    (root/'video.mp4').write_bytes(b'fixture-only')
    state={'status':'pass','case_key':key,'case_key_sha256':identity,'end_to_end_s':10.,
           'video_sha256':hashlib.sha256(b'fixture-only').hexdigest(),'decoded_frames':9,
           'complete_attention_capture':capture}
    stats={'calls':1,'call_records':[{'layer_id':0,'current_start':5,'denoising_pass':0,
            'route_plan_sha256':'other' if changed_route else 'same'}], 'timing':{},'transferred_bytes':100}
    for name,data in [('case_state.json',state),('sparse_history_stats.json',stats),
                      ('case_config.json',{'case_key':key,'latent_frames':3})]:
        (root/name).write_text(json.dumps(data))
    torch.save(torch.ones(1,3,2),root/'latents.pt')


def test_complete_latents_and_ordered_route_gate(tmp_path):
    case(tmp_path/'a')
    case(tmp_path/'b')
    assert compare(tmp_path/'a',tmp_path/'b')['status']=='pass'
    case(tmp_path/'c',changed_route=True)
    assert compare(tmp_path/'a',tmp_path/'c')['status']=='fail'


def test_capture_augmented_timing_is_not_promoted(tmp_path):
    case(tmp_path/'a')
    case(tmp_path/'b',capture=True)
    with pytest.raises(ValueError,match='capture-augmented'):
        compare(tmp_path/'a',tmp_path/'b')
