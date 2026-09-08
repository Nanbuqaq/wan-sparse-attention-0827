#!/usr/bin/env python3
"""Review original/negative native5B controls or privileged1.3B past-text pairs."""
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
from scripts.collect_longlive2_reference import validate


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kind',choices=('native5b','past_text'),required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);cases=[];refs={}
    if args.kind=='native5b':
        for lane,(seed,control) in enumerate(((20260909,None),(20260910,None),(20260909,'duck'),(20260910,'empty'))):
            path=args.root/f'lane{lane}/summary.json';d=json.loads(path.read_text());validate(d,seed,control)
            root=path.parent;latent=torch.load(root/'latents.pt',map_location='cpu',weights_only=True)
            if control is None:refs[seed]=(d['noise_sha256'],latent[:,:80].clone())
            else:
                noise,prefix=refs[seed]
                if d['noise_sha256']!=noise or not torch.equal(prefix,latent[:,:80]):raise ValueError('native negative control changed pre-return trajectory')
            cases.append(dict(id=f'lane{lane}',root=root,seed=seed,control=control,gpu=d['gpu'],latent_frames=128,
                segments=d['segments'],source_summary=str(path),source_summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                runner_commit=d['runner_commit'],upstream_source_commit=d['upstream_source_SHA'],
                prefix_compared_with_original=control is not None,
                prefix_before_return_exact=True if control is not None else None))
    else:
        for seed in (20260909,20260910):
            path=args.root/f'seed{seed}/summary.json';d=json.loads(path.read_text())
            if d['status']!='pass' or len(d['variants'])!=2:raise ValueError('incomplete past text control')
            if {v['variant'] for v in d['variants']}!={'official_recache','privileged_past_text_restatement'}:raise ValueError('wrong text arms')
            first=None
            for row in d['variants']:
                root=path.parent/row['variant'];latent=torch.load(root/'latents.pt',map_location='cpu',weights_only=True)
                if first is None:first=(row['identity']['noise'],latent[:,:78].clone())
                if row['identity']['noise']!=first[0] or not torch.equal(first[1],latent[:,:78]):raise ValueError('text control prefix changed')
                if row['history_H2D_bytes'] or row['archive_bytes']:raise ValueError('past text acquired hidden historyKV')
                cases.append(dict(id=f's{seed}__'+row['variant'],root=root,seed=seed,control=row['variant'],gpu=d['gpu'],
                    latent_frames=120,segments=row['segments_used'],source_summary=str(path),
                    source_summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),runner_commit=d['source_commit'],
                    prefix_before_return_exact=True,privileged_past_text_not_online_method=True,history_KV_bytes=0))
    for case in cases:
        video=case.pop('root')/'video.mp4';label=case['id'];segments=case['segments'];length=case['latent_frames']
        case['video']=str(video)
        case['diagnostic']=analyze(dict(id=label,video=str(video),latent_frames=length,status='pass'),args.output,samples_per_quarter=16)
        frames=decode_frames(video);start=segments[-1]['start_latent'];boards={}
        first_return_end=min(start+(8 if args.kind=='native5b' else 9),length)
        intervals={'source_reveal':(pixel_start(segments[2]['start_latent']-6),pixel_start(segments[2]['start_latent'])),
            'away':(pixel_start(segments[2]['start_latent']),pixel_start(start)),
            'first_return':(pixel_start(start),pixel_start(first_return_end)),
            'late_return':(pixel_start(length-24),len(frames))}
        for part,(a,b) in intervals.items():
            path=args.output/(label+'__'+part+'.png');storyboard(frames,np.linspace(a,b-1,16).round().astype(int),path);boards[part]=str(path)
        case.update(boards=boards,semantic_quality='pending_descriptive_review')
    (args.output/'summary.json').write_text(json.dumps(dict(status='pass',kind=args.kind,cases=cases,
        case_count=len(cases),missing=0,cross_backbone_speed_or_quality_ranking=False),indent=2)+'\n')
    print(json.dumps(dict(status='pass',cases=len(cases),kind=args.kind)))


if __name__=='__main__':main()
