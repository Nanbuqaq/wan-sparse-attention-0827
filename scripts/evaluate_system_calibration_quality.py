#!/usr/bin/env python3
"""Matched development quality, with explicit cross-commit Dense reuse."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_videos import read_video, psnr, ssim, lpips_distances, sha256
from adapters.longlive_sparse.offline_eval import output_error_metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--states',required=True)
    parser.add_argument('--prompt',required=True)
    parser.add_argument('--linear-weights',required=True)
    parser.add_argument('--trunk-weights',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('real CUDA LPIPS evaluation required')
    all_cases=json.loads(Path(args.states).read_text())['cases']
    cases=[case for case in all_cases if case['prompt_id']==args.prompt]
    dense=[case for case in cases if case['method']=='rag_dense' and case['status']=='pass']
    if len(dense)!=1:
        raise ValueError('exactly one audited Dense reference required')
    reference=dense[0]
    video=read_video(Path(reference['video']))
    ref_latent=torch.load(Path(reference['video']).parent/'latents.pt',map_location='cpu',weights_only=True)
    metric=json.loads((ROOT/'configs/quality/lpips_alex_v0p1.json').read_text())
    late_start=3*len(video)//4
    rows=[]
    for case in cases:
        if case['status']!='pass':
            rows.append({'case_id':case['id'],'status':case['status'],'failure_reason':case.get('failure_reason')})
            continue
        for key in ('prompt_sha256','seed','latent_frames','rope_policy'):
            if reference['case_key'][key]!=case['case_key'][key]:
                raise ValueError(f'unmatched quality pair: {key}')
        if case['id']==reference['id']:
            continue
        path=Path(case['video'])
        if sha256(path)!=case['video_sha256']:
            raise ValueError('candidate video hash mismatch')
        candidate=read_video(path)
        if candidate.shape!=video.shape:
            raise ValueError('video shapes differ')
        distances,error,provenance=lpips_distances(video,candidate,
            weights_path=Path(args.linear_weights),expected_sha256=metric['linear_weights']['sha256'],
            expected_version=metric['package_versions']['lpips'],
            trunk_weights_path=Path(args.trunk_weights),expected_trunk_sha256=metric['trunk_weights']['sha256'],
            expected_torch_version=metric['package_versions']['torch'],
            expected_torchvision_version=metric['package_versions']['torchvision'])
        if error or distances is None:
            raise RuntimeError(f'audited LPIPS unavailable: {error}')
        psnrs=[psnr(ref,cur) for ref,cur in zip(video,candidate)]
        ssims=[ssim(ref,cur) for ref,cur in zip(video,candidate)]
        latent=torch.load(path.parent/'latents.pt',map_location='cpu',weights_only=True)
        row={'case_id':case['id'],'status':'pass','method':case['method'],
             'system':case['case_key'].get('system'),'method_params':case['case_key'].get('method_params'),
             'generation_commit':case['case_key']['commit'],'reference_generation_commit':reference['case_key']['commit'],
             'video_sha256':case['video_sha256'],'reference_video_sha256':reference['video_sha256'],
             'decoded_frames':len(video),'history_transfer_density':case.get('history_transfer_density'),
             'latent_error':output_error_metrics(ref_latent,latent),
             'psnr_mean':float(np.mean(psnrs)),'ssim_mean':float(np.mean(ssims)),'lpips_mean':float(np.mean(distances)),
             'late_quarter_psnr_mean':float(np.mean(psnrs[late_start:])),
             'late_quarter_ssim_mean':float(np.mean(ssims[late_start:])),
             'late_quarter_lpips_mean':float(np.mean(distances[late_start:])),
             'lpips_provenance':provenance,
             'timing_scope':case.get('timing_scope'),
             'absolute_subject_state_quality_proven':False}
        rows.append(row)
        print(json.dumps({key:value for key,value in row.items() if key in ('case_id','psnr_mean','ssim_mean','lpips_mean','latent_error')}),flush=True)
        del candidate,latent
    result={'status':'pass','scope':'relative_dense_development_fidelity_not_absolute_quality',
            'prompt':args.prompt,'reference_case':reference['id'],'rows':rows,
            'cross_commit_reuse_explicit':True,'formal_promotion_allowed':False}
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
