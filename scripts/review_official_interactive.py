#!/usr/bin/env python3
"""Source-audited paired official recache review; no inferred quality winner."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_video_review_storyboards import analyze,decode_frames,storyboard
from scripts.review_revisit_videos import pixel_start


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    results=[]
    for lane in range(4):
        root=args.root/f'lane{lane}';source=root/'summary.json';report=json.loads(source.read_text())
        if report['status']!='pass':raise ValueError('incomplete official control lane')
        first_switch=report['segments'][1]['start_latent']
        reference=torch.load(root/'cross_only/latents.pt',map_location='cpu',weights_only=True)
        rows=[]
        for entry in report['variants']:
            name=entry['variant'];case=root/name;code=f'lane{lane}__{name}'
            latent=torch.load(case/'latents.pt',map_location='cpu',weights_only=True)
            if not torch.equal(reference[:,:first_switch],latent[:,:first_switch]):raise ValueError('prefix changed before any prompt switch')
            diagnostics=analyze(dict(id=code,video=str(case/'video.mp4'),latent_frames=120,status='pass'),args.output,samples_per_quarter=16)
            frames=decode_frames(case/'video.mp4');boards={}
            for label,(start,end) in {'reveal':(pixel_start(33),pixel_start(48)),
                'away':(pixel_start(48),pixel_start(78)),'return':(pixel_start(78),len(frames))}.items():
                path=args.output/(code+'__'+label+'.png')
                storyboard(frames,np.linspace(start,end-1,16).round().astype(int),path);boards[label]=str(path)
            rows.append(dict(variant=name,video=str(case/'video.mp4'),boards=boards,diagnostics=diagnostics,
                generation_decode_encode_s=entry['generation_decode_encode_s'],recache_events=entry['recache_events'],
                prefix_before_first_switch_equal=True,history_H2D_bytes=entry['history_H2D_bytes'],archive_bytes=entry['archive_bytes']))
        results.append(dict(lane=lane,seed=report['seed'],negative_control=report['novel_control'],gpu=report['gpu'],
            source_commit=report['source_commit'],upstream_source_sha256=report['upstream_interactive_sha256'],
            source_summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),cases=rows,quality='pending_descriptive_review'))
    (args.output/'summary.json').write_text(json.dumps(dict(status='pass',cases=8,missing=0,lanes=results),indent=2)+'\n')
    print(json.dumps(dict(status='pass',cases=8,prefix_checks=True)))


if __name__=='__main__':main()
