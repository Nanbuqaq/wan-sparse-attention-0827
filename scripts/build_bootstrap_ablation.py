#!/usr/bin/env python3
"""Four diagnostic interventions; keep budget and all other routing tiers fixed."""
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
    params=suite['method_params']['rope_aligned_final_history']
    original=suite['cases'];cases=[]
    for layer in (-1,9):
        for item in original:
            case=copy.deepcopy(item)
            case.update(method_params={'bootstrap_layer':layer},complete_capture=True,
                        ablation='all_layers_single_candidate' if layer==-1 else 'layer9_single_candidate')
            cases.append(case)
    method='rope_bootstrap_ablation_history'
    suite.update(status='frozen_bootstrap_causal_ablation_not_promoted',methods=[method],
        method_params={method:params},method_history_densities={method:.25},cases=cases,
        capture_layers=[0,9],capture_starts=[28080],capture_passes=1)
    expected=[]
    for index,case in enumerate(cases):
        identity=build_case_identity(commit=commit,method=method,prompt_id=case['prompt_id'],prompt=case['prompt'],
            seed=case['seed'],latent_frames=39,history_density=.25,rope_policy='upstream_zero',
            refresh_policy='per_chunk',backend='grouped_fa2',system_identity=suite['longlive_system'],
            method_params={**params,**case['method_params']})
        expected.append({**identity,'method':method,'prompt_id':case['prompt_id'],'seed':case['seed'],
                         'latent_frames':39,'lane':index%2,'ablation':case['ablation']})
    return suite,{'scope':'targeted_bootstrap_causal_ablation','cases':expected}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);args=p.parse_args()
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    suite,expected=build(commit)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    for name,data in [('suite',suite),('expected',expected)]:
        (out/f'{name}.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({'cases':len(expected['cases']),'source':commit}))


if __name__=='__main__':main()
