#!/usr/bin/env python3
"""All128 return RGB frames for both predeclared recent-bound settled cases."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import av
    import numpy as np
    from scripts.build_video_review_storyboards import storyboard
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);rows=[]
    for seed in (20260919,20260920):
        root=args.root/f'seed{seed}'/'recent_virtual';summary=root/'summary.json'
        assert json.loads(summary.read_text())['status']=='pass'
        frames=[]
        with av.open(str(root/'video.mp4')) as container:
            for frame in container.decode(video=0):frames.append(frame.reformat(width=208,height=120).to_ndarray(format='rgb24'))
        assert len(frames)==509
        pages=[]
        for page in range(4):
            output=args.output/f'seed{seed}__return_all_{page}.png'
            indices=np.arange(381+32*page,413+32*page);storyboard(frames,indices,output);pages.append(output.name)
        rows.append(dict(seed=seed,first_pixel=381,last_pixel=508,pages=pages,summary_sha256=hashlib.sha256(summary.read_bytes()).hexdigest()))
    (args.output/'manifest.json').write_text(json.dumps(dict(cases=rows,all_return_frames_rendered=True,review_completed_by_script=False),indent=2)+'\n')
    print(json.dumps(dict(seeds=2,pixel_frames_per_seed=128)))


if __name__=='__main__':main()
