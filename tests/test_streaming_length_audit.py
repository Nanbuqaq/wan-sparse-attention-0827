from copy import deepcopy

import pytest

from scripts.audit_streaming_length_extension import summarize


def test_same_identity_and_each_complete_repeat_required():
    rows=[dict(variant=a,repetition=r,status='pass',exact_reference=True,diagnostic_timing=False,
        decoded_frames=957,identity={'latent':'x','ordered_routes':'y','noise':'n','raw_RGB':'z'},
        generation_decode_encode_s=(10 if i==0 else [8,12,9][r]),sink={'first_packet_muxed_s':1},
        peak_GPU_bytes=4,archive_storage={}) for i,a in enumerate(('batch_current_stream','async_priority_current_stream')) for r in range(3)]
    report=dict(status='pass',latent_frames=240,variants=rows,prompt={},method='Final',gpu='test',source_commit='a'*40)
    assert summarize(report)['negative_repetitions']==1
    bad=deepcopy(report);bad['variants'][2]['identity']['noise']='different'
    with pytest.raises(ValueError):summarize(bad)
    bad=deepcopy(report);bad['variants'][2]['repetition']=1
    with pytest.raises(ValueError):summarize(bad)
