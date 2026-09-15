#!/usr/bin/env python3
"""Extract fixed committed past images from existing private artifacts on CPU."""
import argparse
import hashlib
import json
from pathlib import Path

import av


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cases',required=True,help='Four existing case paths relative to declared input root')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    roots=args.cases.split(',')
    labels=('pattern','close','motion','pattern_second')
    if len(roots)!=4:raise ValueError('four frozen sources required')
    args.output.mkdir(parents=True,exist_ok=False)
    for label,relative in zip(labels,roots):
        case=(args.root/relative).resolve()
        if not case.is_relative_to(args.root.resolve()):raise ValueError('case escapes declared private inputs')
        summary=json.loads((case/'summary.json').read_text())
        if summary['status']!='pass' or summary['latent_shape']!=[1,128,48,44,80]:
            raise ValueError('original-resolution successful native source required')
        prompt=summary['prompts_per_block'][12].removeprefix('The scene transitions. ')
        if any(x!=prompt for x in summary['prompts_per_block'][13:]):
            raise ValueError('this frozen suffix must contain one arrived request')
        image=args.output/(label+'.png')
        with av.open(str(case/'video.mp4')) as video:
            video.streams.video[0].codec_context.thread_count=2
            for i,frame in enumerate(video.decode(video=0)):
                if i==188:frame.to_image().save(image);break
        if not image.exists():raise ValueError('source prefix incomplete')
        spec=dict(schema='causal_visual_restart_v1',label=label,prompt=prompt,image=image.name,
            image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),source_pixel_index=188,
            source_committed_latents=48,request_arrival_latent=96,source_video=str(case/'video.mp4'),
            source_video_sha256=hashlib.sha256((case/'video.mp4').read_bytes()).hexdigest(),
            source_summary_sha256=hashlib.sha256((case/'summary.json').read_bytes()).hexdigest(),
            source_latent_sha256=summary['latent_sha256'],source_seed=summary['seed'],source_hardware=summary['gpu'],
            source_representation='fixed past decoded MP4 RGB; no future image selection',request_reference=str(case/'summary.json'))
        (args.output/(label+'.json')).write_text(json.dumps(spec,indent=2))


if __name__=='__main__':main()
