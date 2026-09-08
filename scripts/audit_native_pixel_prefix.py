#!/usr/bin/env python3
"""Verify full-resolution decoded prefix equivalence before a memory intervention."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import av
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();groups=[]
    for lane in range(2):
        rows=[]
        for arm in ('none','shot','reset_reveal','reset_away'):
            path=args.root/f'lane{lane}'/arm/'video.mp4';digest=hashlib.sha256();count=0
            with av.open(str(path)) as container:
                for frame in container.decode(video=0):
                    if count==381:break
                    rgb=frame.to_ndarray(format='rgb24')
                    if rgb.shape!=(704,1280,3):raise ValueError('not original resolution')
                    digest.update(rgb.tobytes());count+=1
            if count!=381:raise ValueError('prefix video truncated')
            rows.append(dict(arm=arm,frames=count,decoded_RGB_prefix_sha256=digest.hexdigest()))
        groups.append(dict(lane=lane,exact=len({r['decoded_RGB_prefix_sha256'] for r in rows})==1,cases=rows))
    result=dict(status='pass' if all(g['exact'] for g in groups) else 'fail',groups=groups,
                prefix_pixel_interval=[0,381],prefix_latent_interval=[0,96])
    with args.output.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='groups'}))


if __name__=='__main__':main()
