#!/usr/bin/env python3
"""Phase-aligned Dense workload boards and causal retrieval-coverage diagnostics."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_video_review_storyboards import analyze,decode_frames,storyboard


def pixel_start(latent):return 0 if latent==0 else 4*latent-3


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for source in sorted(args.root.glob('lane*/summary.json')):
        report=json.loads(source.read_text())
        if report['status']!='pass':raise ValueError('failed Dense lane cannot be silently reviewed as complete')
        entry=report['variants'][0];root=source.parent/entry['variant']
        segments=report['scenario']['segments'];length=report['latent_frames']
        code=report['scenario']['id']+'__s'+str(report['seed'])
        diagnostic=analyze(dict(id=code,video=str(root/'video.mp4'),latent_frames=length,status='pass'),args.output,samples_per_quarter=16)
        frames=decode_frames(root/'video.mp4');boards=[]
        for i,segment in enumerate(segments):
            start=pixel_start(segment['start_latent'])
            end=pixel_start(segments[i+1]['start_latent']) if i+1<len(segments) else len(frames)
            path=args.output/(code+f'__phase{i}_'+segment['role']+'.png')
            storyboard(frames,np.linspace(start,end-1,16).round().astype(int),path)
            boards.append(dict(segment=i,role=segment['role'],start_pixel=start,end_pixel_exclusive=end,path=str(path)))
        retrieval=json.loads((root/'retrieval.json').read_text())
        ret_start=segments[-1]['start_latent'];phase_counts=Counter();calls=[]
        for call in retrieval:
            if call['query_frame']<ret_start:continue
            counts=Counter()
            for frame in call['selected_global_frames'][0]:
                phase=max(i for i,s in enumerate(segments) if s['start_latent']<=frame)
                counts[segments[phase]['role']]+=1
            phase_counts.update(counts)
            calls.append(dict(query_latent=call['query_frame'],source_phase_counts=dict(counts)))
        rows.append(dict(id=code,status='technical_pass',scenario=report['scenario'],seed=report['seed'],gpu=report['gpu'],
            source_commit=report['source_commit'],summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            video=str(root/'video.mp4'),diagnostics=diagnostic,phase_boards=boards,
            return_history_source_phase_counts=dict(phase_counts),return_retrieval=calls,
            retrieval_counts_are_not_attention_mass=True,semantic_validity='manual_review_pending',
            self_KV_recache_not_used=True))
    if len(rows)!=4:raise ValueError('exactly four frozen Dense workload cases required')
    (args.output/'summary.json').write_text(json.dumps(dict(status='pass',cases=rows,sparse_promotion=False),indent=2)+'\n')
    print(json.dumps(dict(status='pass',cases=4,semantic_validity='pending_manual_review')))


if __name__=='__main__':main()
