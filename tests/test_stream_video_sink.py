import time
import av
import torch

from adapters.longlive_sparse.stream_video_sink import IncrementalVideoSink
from adapters.longlive_sparse.full_flow_profile import normalize_raw_vae, unit_video_to_rgb
from adapters.longlive_sparse.history_cache import tensor_sha256


def test_streamed_canonical_hash_matches_whole_RGB_and_frame_count(tmp_path):
    raw = torch.rand(1, 33, 3, 16, 16, generator=torch.Generator().manual_seed(91))*2-1
    path = tmp_path/'stream.mp4'
    sink = IncrementalVideoSink(path, expected_frames=33, started=time.perf_counter())
    for begin, end in ((0, 9), (9, 21), (21, 33)):
        sink(raw[:, begin:end])
    result = sink.close()
    assert result['raw_RGB_sha256'] == tensor_sha256(unit_video_to_rgb(normalize_raw_vae(raw)))
    assert result['first_packet_muxed_s'] is not None
    with av.open(str(path)) as video:
        assert sum(1 for _ in video.decode(video=0)) == 33


def test_unit_input_is_not_normalized_twice(tmp_path):
    unit=torch.rand(1,5,3,16,16,generator=torch.Generator().manual_seed(21))
    sink=IncrementalVideoSink(tmp_path/'unit.mp4',expected_frames=5,started=time.perf_counter(),input_range='unit')
    sink(unit)
    assert sink.close()['raw_RGB_sha256']==tensor_sha256(unit_video_to_rgb(unit))
