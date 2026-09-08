#!/usr/bin/env python3
"""Full-local baseline review against its own initial identity, not RAG pixels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_video_review_storyboards import analyze,decode_frames,storyboard
from scripts.review_revisit_videos import pixel_start


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-cases',type=int,default=4)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for source in sorted(args.root.glob('*/summary.json')):
        summary=json.loads(source.read_text())
        if summary['status']!='pass' or summary['history_mode']!='local' or len(summary['variants'])!=1:
            raise ValueError('not a complete local-only case')
        entry=summary['variants'][0];case=source.parent/entry['variant']
        stats=json.loads((case/'stats.json').read_text())
        if entry['history_H2D_bytes'] or stats['archive_bytes']:raise ValueError('local baseline unexpectedly stores/transfers old KV')
        code=source.parent.name
        diagnostics=analyze(dict(id=code,video=str(case/'video.mp4'),latent_frames=summary['latent_frames'],status='pass'),
                            args.output,samples_per_quarter=16)
        frames=decode_frames(case/'video.mp4');boards={}
        for name,(begin,end) in {'original_identity':(pixel_start(33),pixel_start(48)),
                                'away':(pixel_start(48),pixel_start(78)),
                                'return':(pixel_start(78),len(frames))}.items():
            path=args.output/(code+'__'+name+'.png')
            storyboard(frames,np.linspace(begin,end-1,16).round().astype(int),path)
            boards[name]=str(path)
        rows.append(dict(id=code,seed=summary['seed'],negative_control=summary.get('novel_control'),gpu=summary['gpu'],
            source_commit=summary['source_commit'],summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            video=str(case/'video.mp4'),history_H2D_bytes=0,archive_bytes=0,
            generation_decode_encode_s=entry['generation_decode_encode_s'],boards=boards,diagnostics=diagnostics,
            semantic_quality='review_against_own_reveal_pending',RAG_pixel_equivalence_not_expected=True))
    if len(rows)!=args.expected_cases:raise ValueError(f'expected {args.expected_cases} local baseline cases, got {len(rows)}')
    (args.output/'summary.json').write_text(json.dumps(dict(status='pass',cases=rows,missing=0),indent=2)+'\n')
    print(json.dumps(dict(status='pass',cases=len(rows),zero_history_archive_and_transfer=True)))


if __name__=='__main__':main()
