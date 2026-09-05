#!/usr/bin/env python3
"""One corrected eight-case calibration after the D2H prototype readiness fix."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.case_identity import build_case_identity
from adapters.longlive_sparse.methods import method_spec
from adapters.longlive_sparse.system_config import LongLiveSystemConfig
from scripts.build_system_routing_calibration_suites import build as original_build


def build(commit):
    suites,_=original_build(holdout_path=ROOT/'configs/formal/system_holdout_prompts.json',
        prompt_path=ROOT/'configs/system/profile_calibration_prompts.json',
        method_params_path=ROOT/'configs/formal/method_params.json',
        candidate_path=ROOT/'configs/system/capture_screened_candidates.json',commit=commit)
    del suites['legacy_final_top_p095']
    expected=[]
    for name,suite in suites.items():
        method=suite['methods'][0]
        system=LongLiveSystemConfig(transfer_layout='exact_compact',staging_mode='persistent_separate',
            cpu_pack_policy='archive_runs',gpu_union_cache='per_chunk',
            gpu_union_cache_budget_mib=4096 if method=='rag_dense' else 768,
            archive_offload='pooled_pageable',host_pinned_budget_mib=128)
        suite['status']='frozen_readiness_corrected_development_calibration'
        suite['repair_scope']='D2H-ready CPU V prototypes; common optimized bounded system; audited input noise'
        for case in suite['cases']:
            case['longlive_system']=system.as_dict()
            params={**suite['method_params'][method],**case.get('method_params',{})}
            identity=build_case_identity(commit=commit,method=method,prompt_id=case['prompt_id'],prompt=case['prompt'],
                seed=case['seed'],latent_frames=39,history_density=case['history_density'],
                rope_policy=case['rope_policy'],refresh_policy=case['refresh_policy'],backend=case['backend'],
                system_identity=system.identity_dict(),method_params=params)
            expected.append({**identity,'method':method,'routing_stage':method_spec(method).routing_stage,
                'prompt_id':case['prompt_id'],'seed':case['seed'],'latent_frames':39,'calibration_config_id':name})
    assert len(expected)==8 and len({case['case_key_sha256'] for case in expected})==8
    return suites,{'cases':expected,'scope':'corrected_development_not_formal',
                   'supersedes_method_selection_from':'4b6f976','prior_artifacts_preserved':True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    suites,expected=build(commit)
    root=Path(args.output_dir)
    root.mkdir(parents=True,exist_ok=True)
    for name,suite in suites.items():
        (root/f'suite_{name}.json').write_text(json.dumps(suite,indent=2)+'\n')
    (root/'expected.json').write_text(json.dumps(expected,indent=2)+'\n')
    print(json.dumps({'status':'pass','new_cases':8,'configs':list(suites)}))


if __name__=='__main__':
    main()
