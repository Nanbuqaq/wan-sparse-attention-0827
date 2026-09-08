#!/usr/bin/env python3
"""Full native cut screen, including every pixel frame in the final-away window."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.collect_native_cut_screen import DESIGN,validate
from scripts.build_video_review_storyboards import analyze,decode_frames,storyboard
from scripts.review_revisit_videos import pixel_start


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);rows=[]
    for lane,(scenario,seed) in enumerate(DESIGN):
        source=args.root/f'lane{lane}/summary.json';d=json.loads(source.read_text());validate(d,scenario,seed)
        video=source.parent/'video.mp4';label=f'lane{lane}';frames=decode_frames(video)
        if len(frames)!=509:raise ValueError('full native cut decode count differs')
        diagnostic=analyze(dict(id=label,video=str(video),latent_frames=128,status='pass'),args.output,samples_per_quarter=16)
        boards={}
        for role,(start,end) in {'initial':(0,24),'before_away':(42,48),'away':(48,96),
            'first_return':(96,104),'late_return':(112,128)}.items():
            a,b=pixel_start(start),pixel_start(end);path=args.output/(label+'__'+role+'.png')
            storyboard(frames,np.linspace(a,b-1,16).round().astype(int),path);boards[role]=str(path)
        # Exact128-frame final-away interval, not only sparse contact-sheet samples.
        indices=np.arange(pixel_start(64),pixel_start(96));assert len(indices)==128
        for half,part in enumerate(np.array_split(indices,2)):
            path=args.output/(label+f'__away_tail_all_frames_half{half}.png')
            storyboard(frames,part,path);boards[f'away_tail_all_frames_half{half}']=str(path)
        rows.append(dict(lane=lane,scenario=scenario,seed=seed,gpu=d['gpu'],source_commit=d['runner_commit'],
            source_summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),video=str(video),boards=boards,
            diagnostic=diagnostic,observed_pin_events=d['native_shot_pin_events'],
            final_away_latent_interval=[64,96],all_final_away_pixel_frames_indexed=True,
            scene_and_target_feasibility='pending_assistant_visual_review',
            memory_quality='pending_own_generated_identity_or_state_comparison',Dense_only=True))
    (args.output/'summary.json').write_text(json.dumps(dict(status='pass',cases=4,missing=0,rows=rows,
        semantic_pass_not_inferred_from_cut_marker=True),indent=2)+'\n')
    print(json.dumps(dict(status='pass',cases=4,semantic_review='pending')))


if __name__=='__main__':main()
