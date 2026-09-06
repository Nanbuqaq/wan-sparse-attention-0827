#!/usr/bin/env python3
"""Nine-case independent-seed exploratory replication, three same-GPU triplets."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_aligned_final_probe import build as base_build
from adapters.longlive_sparse.case_identity import build_case_identity


def build(commit, *, candidate='rope_aligned_final_history', seed_base=20260911, latent_frames=39):
    suite,_=base_build(commit)
    if candidate not in ('rope_aligned_final_history','rope_bootstrap_ablation_history') or latent_frames not in (39,120):
        raise ValueError('unsupported frozen replication candidate/length')
    if candidate!='rope_aligned_final_history':
        suite['methods'][-1]=candidate
        params=suite['method_params'].pop('rope_aligned_final_history')
        suite['method_params'][candidate]={**params,'bootstrap_layer':-1}
        suite['method_history_densities'][candidate]=suite['method_history_densities'].pop('rope_aligned_final_history')
    by_prompt={c['prompt_id']:c for c in suite['cases']}
    cases=[]
    for prompt,seed in [('calibration_motion',seed_base),('calibration_state',seed_base),('calibration_state',seed_base+1)]:
        case=copy.deepcopy(by_prompt[prompt]);case.update(seed=seed,latent_frames=latent_frames);cases.append(case)
    suite.update(status='frozen_independent_seed_exploration_no_formal_promotion',cases=cases,
        hypothesis='replicate motion/state interaction and prioritize a second independent state seed',
        pooled_cross_category_mean_allowed=False,formal_promotion_allowed=False)
    expected=[]
    for method in suite['methods']:
        for lane,case in enumerate(cases):
            expected.append({**build_case_identity(commit=commit,method=method,prompt_id=case['prompt_id'],prompt=case['prompt'],
                seed=case['seed'],latent_frames=latent_frames,history_density=suite['method_history_densities'][method],
                rope_policy=suite['rope_policy'],refresh_policy=suite['refresh_policy'],backend=suite['backend'],
                system_identity=suite['longlive_system'],method_params=suite['method_params'][method]),
                'method':method,'prompt_id':case['prompt_id'],'seed':case['seed'],'latent_frames':latent_frames,'lane':lane})
    return suite,{'scope':'nine_case_seed_replication_not_formal_holdouts','cases':expected}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',required=True)
    parser.add_argument('--candidate',default='rope_aligned_final_history')
    parser.add_argument('--seed-base',type=int,default=20260911)
    parser.add_argument('--latent-frames',type=int,default=39)
    args=parser.parse_args()
    source=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    suite,expected=build(source,candidate=args.candidate,seed_base=args.seed_base,latent_frames=args.latent_frames)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    for name,data in [('suite',suite),('expected',expected)]:
        (out/f'{name}.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({'expected':len(expected['cases']),'source':source}))


if __name__=='__main__':main()
