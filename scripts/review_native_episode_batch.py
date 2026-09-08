#!/usr/bin/env python3
"""Own-source review of the episode admission/storage factorial; lanes isolated."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.collect_native_episode_batch import MODES,validate_group
from scripts.build_video_review_storyboards import analyze,decode_frames,storyboard
from scripts.review_revisit_videos import pixel_start


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);groups=[]
    for lane in range(4):
        root=args.root/f'lane{lane}';reports={};errors=[]
        for mode in MODES:
            try:reports[mode]=json.loads((root/mode/'summary.json').read_text())
            except (OSError,ValueError) as error:errors.append(f'{mode}: {error}')
        try:contract=validate_group(reports)
        except (ValueError,KeyError) as error:contract=dict(status='fail',error=str(error));errors.append(str(error))
        rows=[];reference=None
        if reports.get('none',{}).get('status')=='pass':
            reference=torch.load(root/'none/latents.pt',map_location='cpu',weights_only=True)
        for mode in MODES:
            d=reports.get(mode,{})
            if d.get('status')!='pass':
                rows.append(dict(mode=mode,status='fail',error=d.get('traceback','missing terminal success')));continue
            case=root/mode;video=case/'video.mp4';label=f'lane{lane}__{mode}'
            latent=torch.load(case/'latents.pt',map_location='cpu',weights_only=True)
            prefix_equal=reference is not None and torch.equal(reference[:,:96],latent[:,:96])
            frames=decode_frames(video)
            if len(frames)!=509:raise ValueError('full episode video truncated')
            boards={};diagnostic=None
            if mode!='log_reveal' or contract['status']!='pass':
                diagnostic=analyze(dict(id=label,video=str(video),latent_frames=128,status='pass'),args.output,samples_per_quarter=16)
                for part,(a,b) in {'source':(42,48),'first_return':(96,104),'late_return':(112,128)}.items():
                    path=args.output/(label+'__'+part+'.png')
                    storyboard(frames,np.linspace(pixel_start(a),pixel_start(b)-1,16).round().astype(int),path);boards[part]=str(path)
            rows.append(dict(mode=mode,status='technical_pass',video=str(video),decoded_frames=509,
                prefix_before_return_exact=prefix_equal,boards=boards,diagnostic=diagnostic,
                same_pixels_as_raw_reveal=mode=='log_reveal' and contract['status']=='pass',
                summary_sha256=hashlib.sha256((case/'summary.json').read_bytes()).hexdigest(),
                ledger=d['episode_memory']['ledger'],native_DiT_s=d['native_DiT_s'],native_VAE_s=d['native_VAE_s'],
                generation_peak_allocated_bytes=d['generation_peak_allocated_bytes'],
                semantic_quality='pending_assistant_own_identity_state_review'))
        groups.append(dict(lane=lane,contract=contract,errors=errors,cases=rows,
            comparative_attribution_allowed=contract['status']=='pass'))
    result=dict(status='pass' if all(g['contract']['status']=='pass' for g in groups) else 'partial',groups=groups,
        cases=16,failed_lanes_isolated=True,privileged_admission_not_online_method=True,
        correct_quality_not_inferred_from_equivalence=True)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='groups'}))


if __name__=='__main__':main()
