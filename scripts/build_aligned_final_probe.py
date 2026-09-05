#!/usr/bin/env python3
"""Six matched-hardware development cases, three methods per loaded GPU lane."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.case_identity import build_case_identity
from adapters.longlive_sparse.system_config import LongLiveSystemConfig


def build(commit):
    prompts_path=ROOT/'configs/system/profile_calibration_prompts.json'
    prompts=json.loads(prompts_path.read_text())['candidates']
    methods=['rag_dense','transfer_vaware_hybrid_history','rope_aligned_final_history']
    params=json.loads((ROOT/'configs/formal/method_params.json').read_text())['method_params']['transfer_vaware_hybrid_history']
    densities=dict(zip(methods,[1.,.25,.25]))
    system=LongLiveSystemConfig(transfer_layout='exact_compact',staging_mode='persistent_separate',
        cpu_pack_policy='archive_runs',gpu_union_cache='per_chunk',gpu_union_cache_budget_mib=4096,
        archive_offload='pooled_pageable',host_pinned_budget_mib=128)
    suite={'status':'frozen_aligned_final_local_matched_development_probe','experiment_commit':commit,
        'formal_prompts_used':False,'methods':methods,'method_params':{methods[0]:{},methods[1]:params,methods[2]:params},
        'method_history_densities':densities,'backend':'grouped_fa2','history_density':.25,
        'refresh_policy':'per_chunk','rope_policy':'upstream_zero','record_per_call':True,
        'longlive_system':system.as_dict(),
        'cases':[{**p,'seed':20260904,'latent_frames':39,'complete_capture':False} for p in prompts],
        'source_prompts_sha256':hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
        'same_hardware_controls_reason':'candidate-order numerical sensitivity; old H controls are not directly mixed with 4090 quality',
        'formal_promotion_allowed':False}
    expected=[]
    for method in methods:
        for lane,case in enumerate(suite['cases']):
            identity=build_case_identity(commit=commit,method=method,prompt_id=case['prompt_id'],prompt=case['prompt'],
                seed=case['seed'],latent_frames=39,history_density=densities[method],rope_policy='upstream_zero',
                refresh_policy='per_chunk',backend='grouped_fa2',system_identity=system.identity_dict(),
                method_params=suite['method_params'][method])
            expected.append({**identity,'method':method,'prompt_id':case['prompt_id'],'lane':lane,
                             'latent_frames':39,'seed':20260904})
    assert len(expected)==6 and len({r['case_key_sha256'] for r in expected})==6
    return suite,{'scope':'local_matched_development_not_formal','cases':expected}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    suite,expected=build(commit)
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=False)
    (out/'suite.json').write_text(json.dumps(suite,indent=2)+'\n')
    (out/'expected.json').write_text(json.dumps(expected,indent=2)+'\n')
    print(json.dumps({'status':'pass','expected_cases':6,'commit':commit}))


if __name__=='__main__':
    main()
