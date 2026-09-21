import json
import av
import torch
from adapters.longlive_sparse.stream_video_sink import IncrementalVideoSink

def test_instrumentation_preserves_rgb_packets_and_frame_order(tmp_path):
    torch.manual_seed(123)
    raw=torch.rand(1,9,3,32,32)*2-1
    hashes=[];videos=[]
    for enabled in (False,True):
        p=tmp_path/f'{enabled}.mp4'
        sink=IncrementalVideoSink(p,expected_frames=9,started=0,measure_stages=enabled)
        sink(raw[:,:1]);sink(raw[:,1:5]);sink(raw[:,5:])
        hashes.append(sink.close()['raw_RGB_sha256'])
        with av.open(str(p)) as c:
            videos.append([f.to_ndarray(format='rgb24').tobytes() for f in c.decode(video=0)])
        if enabled:
            assert {r['name'] for r in sink.stage_records}=={'pixel_normalize_and_rgb','rgb_hash','frame_construction','encode','mux'}
            assert len([r for r in sink.stage_records if r['name']=='encode'])==9
            assert all(r['wall_s']>=0 and r['caller_thread_cpu_s']>=0 for r in sink.stage_records)
        else:assert sink.stage_records==[]
    assert hashes[0]==hashes[1] and videos[0]==videos[1] and len(videos[0])==9
