#!/usr/bin/env python3
"""Audited phase boards for the privileged pre/post snapshot experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.collect_episode_snapshot import validate
from scripts.build_video_review_storyboards import analyze,decode_frames,storyboard
from scripts.review_revisit_videos import pixel_start


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);groups=[]
    for lane,seed in enumerate((20260909,20260910)):
        root=args.root/f'lane{lane}';path=root/'summary.json';report=json.loads(path.read_text());validate(report,seed)
        reference=torch.load(root/'official_recache/latents.pt',map_location='cpu',weights_only=True);rows=[]
        for row in report['variants']:
            name=row['variant'];case=root/name;label=f's{seed}__{name}'
            latent=torch.load(case/'latents.pt',map_location='cpu',weights_only=True)
            if not torch.equal(latent[:,:78],reference[:,:78]):raise ValueError('pre-return prefix differs')
            diagnostic=analyze(dict(id=label,video=str(case/'video.mp4'),latent_frames=120,status='pass'),args.output,samples_per_quarter=16)
            frames=decode_frames(case/'video.mp4');boards={}
            for part,(start,end) in {'source_snapshot':(pixel_start(42),pixel_start(48)),
                'away':(pixel_start(48),pixel_start(78)),
                'first_return':(pixel_start(78),pixel_start(87)),
                'late_return':(pixel_start(96),len(frames))}.items():
                image=args.output/(label+'__'+part+'.png')
                storyboard(frames,np.linspace(start,end-1,16).round().astype(int),image);boards[part]=str(image)
            rows.append(dict(variant=name,status='technical_pass',video=str(case/'video.mp4'),boards=boards,
                diagnostic=diagnostic,history_H2D_bytes=row['history_H2D_bytes'],archive_bytes=row['archive_bytes'],
                generation_decode_encode_s=row['generation_decode_encode_s'],pre_return_prefix_equal=True,
                semantic_quality='pending_descriptive_review'))
        groups.append(dict(seed=seed,gpu=report['gpu'],source_commit=report['source_commit'],cases=rows,
            source_summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (args.output/'summary.json').write_text(json.dumps(dict(status='pass',cases=6,missing=0,
        privileged_diagnostic_not_online_method=True,groups=groups),indent=2)+'\n')
    print(json.dumps(dict(status='pass',cases=6,semantic_quality='pending_review')))


if __name__=='__main__':main()
