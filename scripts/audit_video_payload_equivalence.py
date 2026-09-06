#!/usr/bin/env python3
"""Separate MP4 container bytes, elementary packets and decoded pixel equality."""
import argparse
import hashlib
import json
from pathlib import Path
from itertools import zip_longest

import av
import numpy as np


def stream_audit(path):
    packets = hashlib.sha256()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        info = {'container_metadata': dict(container.metadata), 'stream_metadata': dict(stream.metadata),
            'codec': stream.codec_context.name, 'pixel_format': str(stream.codec_context.format),
            'extradata_sha256': hashlib.sha256(stream.codec_context.extradata or b'').hexdigest(),
            'time_base': str(stream.time_base), 'duration': stream.duration, 'frame_count': stream.frames}
        for packet in container.demux(stream):
            packets.update(bytes(packet))
    info['packet_payload_sha256'] = packets.hexdigest()
    return info


def compare(left, right):
    a, b = stream_audit(left), stream_audit(right)
    ah, bh = hashlib.sha256(), hashlib.sha256()
    changed, max_abs, squared, components, frames = [], 0, 0., 0, 0
    with av.open(str(left)) as ca, av.open(str(right)) as cb:
        ca.streams.video[0].codec_context.thread_count = 2
        cb.streams.video[0].codec_context.thread_count = 2
        for i, (fa, fb) in enumerate(zip_longest(ca.decode(video=0), cb.decode(video=0))):
            if fa is None or fb is None:
                raise ValueError('different decoded frame counts')
            pa, pb = fa.to_ndarray(format='rgb24'), fb.to_ndarray(format='rgb24')
            if pa.shape != pb.shape:
                raise ValueError('different decoded pixel geometry')
            ah.update(pa.tobytes())
            bh.update(pb.tobytes())
            delta = pa.astype(np.int16)-pb.astype(np.int16)
            local = int(np.abs(delta).max())
            if local:
                changed.append(i)
                max_abs = max(max_abs, local)
            squared += float(np.square(delta.astype(np.float64)).sum())
            components += pa.size
            frames += 1
    return {'left_stream': a, 'right_stream': b, 'frames': frames,
        'packet_payload_equal': a['packet_payload_sha256'] == b['packet_payload_sha256'],
        'decoded_rgb_equal': not changed, 'changed_decoded_frames': changed,
        'decoded_rgb_max_abs_u8': max_abs, 'decoded_rgb_rmse_u8': (squared/components)**.5,
        'left_decoded_rgb_sha256': ah.hexdigest(), 'right_decoded_rgb_sha256': bh.hexdigest()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--states', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    cases = json.loads(Path(args.states).read_text())['cases']
    rows = []
    for lane in range(4):
        pair = [c for c in cases if c['lane'] == lane]
        left = next(c for c in pair if c['case_key']['system']['gpu_union_cache'] == 'per_chunk')
        right = next(c for c in pair if c['case_key']['system']['gpu_union_cache'] == 'hierarchical')
        result = compare(Path(left['video']), Path(right['video']))
        result.update(lane=lane, method=left['method'], prompt=left['prompt_id'],
            left_video_sha256=left['video_sha256'], right_video_sha256=right['video_sha256'])
        rows.append(result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('left_stream', 'right_stream', 'changed_decoded_frames')}), flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as handle:
        json.dump({'status': 'pass', 'scope': 'container_vs_packet_vs_decoded_pixel_audit', 'pairs': rows}, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
