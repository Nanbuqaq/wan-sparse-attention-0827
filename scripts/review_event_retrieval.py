#!/usr/bin/env python3
"""Method-masked event-return and negative-control boards with exact prefix audit."""
import argparse
import hashlib
import json
from pathlib import Path
import random
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
    reports=[];masked=[];rng=random.Random(310620)
    for lane in range(4):
        root=args.root/f'lane{lane}';source=root/'summary.json';report=json.loads(source.read_text())
        if report['status']!='pass':raise ValueError('incomplete event lane')
        baseline=torch.load(root/'scheduled_Dense/latents.pt',map_location='cpu',weights_only=True)
        entries=list(report['variants']);rng.shuffle(entries);rows=[]
        for index,entry in enumerate(entries):
            name=entry['variant'];case=root/name;code=f'lane{lane}_masked{index}'
            latent=torch.load(case/'latents.pt',map_location='cpu',weights_only=True)
            audit=entry.get('event_retrieval');boundary=audit['first_override_latent'] if audit else 120
            if boundary is None:boundary=120
            if not torch.equal(baseline[:,:boundary],latent[:,:boundary]):raise ValueError('causal prefix failed')
            diagnostics=analyze(dict(id=code,video=str(case/'video.mp4'),latent_frames=120,status='pass'),args.output,samples_per_quarter=16)
            frames=decode_frames(case/'video.mp4');boards={}
            for label,(begin,end) in {'original_identity':(pixel_start(33),pixel_start(48)),
                                     'away':(pixel_start(48),pixel_start(78)),
                                     'return':(pixel_start(78),len(frames))}.items():
                path=args.output/(code+'__'+label+'.png')
                storyboard(frames,np.linspace(begin,end-1,16).round().astype(int),path)
                boards[label]=str(path)
            row=dict(variant=name,masked_id=code,video=str(case/'video.mp4'),prefix_equal_before_latent=boundary,
                history_H2D_bytes=entry['history_H2D_bytes'],generation_decode_encode_s=entry['generation_decode_encode_s'],
                event_retrieval=audit,boards=boards,diagnostics=diagnostics)
            rows.append(row)
            masked.append(dict(id=code,seed=report['seed'],negative_control=report.get('novel_control'),
                last_instruction=report['scenario']['segments'][-1]['prompt'],boards=boards))
        reports.append(dict(lane=lane,seed=report['seed'],negative_control=report.get('novel_control'),gpu=report['gpu'],
            source_commit=report['source_commit'],summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),cases=rows,
            visual_outcome='method_name_masked_AI_review_pending_not_independent_human'))
    (args.output/'summary.json').write_text(json.dumps(dict(status='pass',cases=10,missing=0,reports=reports,
        automatic_quality_success=False),indent=2)+'\n')
    (args.output/'masked_manifest.json').write_text(json.dumps(masked,indent=2)+'\n')
    print(json.dumps(dict(status='pass',cases=10,prefix_audited=True,visual_review='pending')))


if __name__=='__main__':main()
