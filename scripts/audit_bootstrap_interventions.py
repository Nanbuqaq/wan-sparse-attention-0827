#!/usr/bin/env python3
"""Verify that targeted videos changed the intended causal decision, not inputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.memory_dynamics import compare_coordinates
from adapters.longlive_sparse.route_plan import HistoryRoutePlan


def main():
    p=argparse.ArgumentParser();p.add_argument('--ablation-root',required=True);p.add_argument('--reference-root',required=True)
    p.add_argument('--capture-root',required=True);p.add_argument('--plan-root',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(2)
    root=Path(args.ablation_root)
    states=json.loads((root/'states.json').read_text())['cases']
    references=json.loads((Path(args.reference_root)/'states.json').read_text())['cases']
    captures=json.loads((Path(args.capture_root)/'trajectory_audit.json').read_text())['cases']
    rows=[]
    for case in states:
        raw=next(r for r in references if r['prompt_id']==case['prompt_id'] and r['method']=='transfer_vaware_hybrid_history')
        aligned=next(r for r in references if r['prompt_id']==case['prompt_id'] and r['method']=='rope_aligned_final_history')
        cutoff=case['case_key']['method_params']['bootstrap_layer']
        latent=torch.load(Path(case['video']).parent/'latents.pt',map_location='cpu',weights_only=True)
        ref_raw=torch.load(Path(raw['video']).parent/'latents.pt',map_location='cpu',weights_only=True)
        ref_phase=torch.load(Path(aligned['video']).parent/'latents.pt',map_location='cpu',weights_only=True)
        checks={'same_noise':case['initial_noise_sha256']==raw['initial_noise_sha256']==aligned['initial_noise_sha256'],
                'same_pre_intervention_latents':torch.equal(latent[:,:18],ref_phase[:,:18])}
        if cutoff==-1:checks['first_sparse_chunk_exact_raw_control']=torch.equal(latent[:,:21],ref_raw[:,:21])
        own_dir=Path(case['video']).parent.parent/'complete_attention_captures'/case['id']
        changed=0 if cutoff==-1 else 9
        reference_method='transfer_vaware_hybrid_history' if cutoff==-1 else 'rope_aligned_final_history'
        source=next(r for r in captures if r['prompt']==case['prompt_id'] and r['method']==reference_method)
        name=f'layer{changed:02d}_start00028080_pass00.pt'
        current=torch.load(own_dir/name,map_location='cpu',weights_only=True)
        previous=torch.load(Path(source['capture_dir'])/name,map_location='cpu',weights_only=True)
        for field in ('query','query_unrotated','exact_key','exact_value','key','key_unrotated','value','frame_ids','token_ids'):
            checks[f'first_changed_inputs_{field}_exact']=torch.equal(current[field],previous[field])
        task=f'{reference_method}__{case["prompt_id"]}__L{changed:02d}__start28080.pt'
        plan_path=Path(args.plan_root)/case['prompt_id'].removeprefix('calibration_')/task
        planned=torch.load(plan_path,map_location='cpu',weights_only=True)['plans']['transfer_vaware_hybrid_history']
        a,b=HistoryRoutePlan.from_state_dict(current['route_plan']),HistoryRoutePlan.from_state_dict(planned)
        checks['planned_raw_coordinates_executed']=compare_coordinates(a,b,token_base=1560)['jaccard']==1.
        rows.append({'id':case['id'],'prompt':case['prompt_id'],'bootstrap_layer':cutoff,'checks':checks,
                     'status':'pass' if all(checks.values()) else 'fail'})
    result={'status':'pass' if len(rows)==4 and all(r['status']=='pass' for r in rows) else 'fail',
            'scope':'first_changed_input_and_selection_causal_control','cases':rows,'formal_promotion_allowed':False}
    with Path(args.output).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
    if result['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
