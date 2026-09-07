"""Incremental CPU video sink with canonical RGB hashing and packet timestamps."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

import torch

from .full_flow_profile import normalize_raw_vae, unit_video_to_rgb
from .profiling import profiled


class IncrementalVideoSink:
    def __init__(self, path, *, expected_frames, started, fps=16):
        self.path, self.expected_frames, self.started = Path(path), expected_frames, started
        self.fps = fps
        self.container = self.stream = None
        self.frames = 0
        self.first_pixels_s = self.first_packet_muxed_s = None
        self.sha = hashlib.sha256()
        self.closed = False
        self.shape = None

    @profiled('video/incremental_encode')
    def __call__(self, raw_pixels, record=None):
        import av
        if self.closed or raw_pixels.ndim != 5 or raw_pixels.shape[0] != 1:
            raise ValueError('sink requires an open batch-one stream')
        rgb = unit_video_to_rgb(normalize_raw_vae(raw_pixels))
        if self.first_pixels_s is None:
            self.first_pixels_s = time.perf_counter()-self.started
        if self.container is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                raise ValueError('preserve existing video')
            self.container = av.open(str(self.path), mode='w', options={'movflags': 'frag_keyframe+empty_moov+default_base_moof'})
            self.stream = self.container.add_stream('libx264', rate=self.fps)
            self.stream.width, self.stream.height = rgb.shape[3], rgb.shape[2]
            self.stream.pix_fmt = 'yuv420p'
            self.stream.codec_context.thread_count = 2
            self.stream.options = {'preset': 'veryfast', 'crf': '18', 'tune': 'zerolatency'}
            self.shape = (1, self.expected_frames, rgb.shape[2], rgb.shape[3], 3)
            self.sha.update(str(rgb.dtype).encode())
            self.sha.update(json.dumps(list(self.shape)).encode())
        if tuple(rgb.shape[2:]) != self.shape[2:]:
            raise ValueError('stream RGB geometry changed')
        self.sha.update(rgb.contiguous().view(torch.uint8).numpy().tobytes())
        for value in rgb[0]:
            frame = av.VideoFrame.from_ndarray(value.contiguous().numpy(), format='rgb24')
            frame.pts, frame.time_base = self.frames, Fraction(1, self.fps)
            for packet in self.stream.encode(frame):
                self.container.mux(packet)
                if self.first_packet_muxed_s is None:
                    self.first_packet_muxed_s = time.perf_counter()-self.started
            self.frames += 1

    def close(self):
        if self.closed:
            raise RuntimeError('sink already closed')
        self.closed = True
        if self.container is not None:
            try:
                for packet in self.stream.encode(None):
                    self.container.mux(packet)
                    if self.first_packet_muxed_s is None:
                        self.first_packet_muxed_s = time.perf_counter()-self.started
            finally:
                self.container.close()
        if self.frames != self.expected_frames:
            raise RuntimeError(f'stream encoded {self.frames} frames, expected {self.expected_frames}')
        return dict(frames=self.frames, raw_RGB_sha256=self.sha.hexdigest(), first_RGB_ready_s=self.first_pixels_s,
            first_packet_muxed_s=self.first_packet_muxed_s, encoder='libx264_veryfast_crf18_zerolatency_threads2',
            muxer='fragmented_mp4', first_client_display_time_measured=False)
