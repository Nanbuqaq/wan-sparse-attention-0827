#!/usr/bin/env python3
"""Encode identical saved RGB repeatedly using local or pinned shared PyAV."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--av-site')
    p.add_argument('--frames', type=int, default=81)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.av_site:
        sys.path.insert(0, args.av_site)
    import av
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    frames = []
    with av.open(args.video) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format='rgb24'))
            if len(frames) == args.frames:
                break
    rgb = np.stack(frames)
    source_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
    results = []
    for mode in ('default', 'single_thread'):
        for repeat in range(3):
            path = out/f'{mode}_{repeat}.mp4'
            start = time.perf_counter()
            with av.open(str(path), 'w') as container:
                stream = container.add_stream('libx264', rate=16)
                stream.width, stream.height = rgb.shape[2], rgb.shape[1]
                stream.pix_fmt = 'yuv420p'
                if mode == 'single_thread':
                    stream.options = {'threads': '1', 'x264-params': 'threads=1:lookahead-threads=1:sliced-threads=0'}
                for pixels in rgb:
                    frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
                    frame.pict_type = av.video.frame.PictureType.NONE
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            encoded_s = time.perf_counter()-start
            decoded = hashlib.sha256()
            with av.open(str(path)) as container:
                for frame in container.decode(video=0):
                    decoded.update(frame.to_ndarray(format='rgb24').tobytes())
            with av.open(str(path)) as container:
                sei = re.search(rb'x264[^\x00]{0,1600}', bytes(next(container.demux(video=0))))
            results.append({'mode': mode, 'repeat': repeat, 'encode_s': encoded_s,
                'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'decoded_rgb_sha256': decoded.hexdigest(), 'encoder_sei': sei.group().decode('latin1') if sei else None})
            print(json.dumps({k:v for k,v in results[-1].items() if k != 'encoder_sei'}), flush=True)
    result = {'status': 'pass', 'scope': 'identical_RGB_repeated_encoding_only_no_VAE_or_generation',
        'input_rgb_sha256': source_sha, 'frames': len(rgb), 'pyav_version': av.__version__, 'pyav_path': av.__file__,
        'library_versions': av.library_versions, 'records': results,
        'modes': {mode: {'unique_file_shas': len({r['file_sha256'] for r in results if r['mode'] == mode}),
                        'unique_decoded_rgb_shas': len({r['decoded_rgb_sha256'] for r in results if r['mode'] == mode})}
                  for mode in ('default', 'single_thread')}}
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
