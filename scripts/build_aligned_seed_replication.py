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


def build(commit):
    suite,_=base_build(commit)
    by_prompt={c['prompt_id']:c for c in suite['cases']}
    cases=[]
    for prompt,seed in [('calibration_motion',20260911),('calibration_state',20260911),('calibration_state',20260912)]:
        case=copy.deepcopy(by_prompt[prompt]);case['seed']=seed;cases.append(case)
    suite.update(status='frozen_independent_seed_exploration_no_formal_promotion',cases=cases,
        hypothesis='replicate motion/state interaction and prioritize a second independent state seed',
        pooled_cross_category_mean_allowed=False,formal_promotion_allowed=False)
    expected=[]
    for method in suite['methods']:
        for lane,case in enumerate(cases):
            expected.append({**build_case_identity(commit=commit,method=method,prompt_id=case['prompt_id'],prompt=case['prompt'],
                seed=case['seed'],latent_frames=39,history_density=suite['method_history_densities'][method],
                rope_policy=suite['rope_policy'],refresh_policy=suite['refresh_policy'],backend=suite['backend'],
                system_identity=suite['longlive_system'],method_params=suite['method_params'][method]),
                'method':method,'prompt_id':case['prompt_id'],'seed':case['seed'],'latent_frames':39,'lane':lane})
    return suite,{'scope':'nine_case_seed_replication_not_formal_holdouts','cases':expected}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',required=True);args=parser.parse_args()
    source=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    suite,expected=build(source)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    for name,data in [('suite',suite),('expected',expected)]:
        (out/f'{name}.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({'expected':len(expected['cases']),'source':source}))


if __name__=='__main__':main()
