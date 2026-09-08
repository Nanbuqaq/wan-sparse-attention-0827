#!/usr/bin/env python3
"""Audit the paired past-episode intervention without calling it an online method."""
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
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    reports=[]
    for seed in (20260909,20260910):
        root=args.root/f'seed{seed}';source=root/'summary.json';summary=json.loads(source.read_text())
        if summary['status']!='pass' or len(summary['variants'])!=3:raise ValueError('all three anchor arms required')
        reference=torch.load(root/'scheduled_Dense/latents.pt',map_location='cpu',weights_only=True)
        rows=[]
        for entry in summary['variants']:
            name=entry['variant'];case=root/name;latent=torch.load(case/'latents.pt',map_location='cpu',weights_only=True)
            if not torch.equal(reference[:,:78],latent[:,:78]):raise ValueError('pre-intervention prefix changed')
            code=f's{seed}__{name}'
            diagnostic=analyze(dict(id=code,video=str(case/'video.mp4'),latent_frames=120,status='pass'),args.output,samples_per_quarter=16)
            frames=decode_frames(case/'video.mp4')
            intervals={'original_identity':(pixel_start(33),pixel_start(48)),
                       'first_return_chunk':(pixel_start(78),pixel_start(81)),
                       'entire_return':(pixel_start(78),len(frames))}
            boards={}
            for label,(start,end) in intervals.items():
                path=args.output/(code+'__'+label+'.png')
                storyboard(frames,np.linspace(start,end-1,16).round().astype(int),path)
                boards[label]=str(path)
            rows.append(dict(variant=name,video=str(case/'video.mp4'),prefix_latents_bitwise_equal=True,
                history_H2D_bytes=entry['history_H2D_bytes'],generation_decode_encode_s=entry['generation_decode_encode_s'],
                intervention=entry['anchor_probe'],diagnostics=diagnostic,boards=boards))
        if len({r['history_H2D_bytes'] for r in rows})!=1:
            raise ValueError('anchor intervention changed total raw history transport budget')
        reports.append(dict(seed=seed,gpu=summary['gpu'],source_commit=summary['source_commit'],
            summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),cases=rows,
            same_prefix_and_history_bytes=True,visual_outcome='pending_descriptive_review'))
    result=dict(status='pass',cases=6,missing=0,reports=reports,privileged_past_interval_teacher=True,
        automatic_online_method=False,visual_identity_success_not_inferred=True)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='pass',cases=6,same_prefix_and_history_bytes=True)))


if __name__=='__main__':main()
