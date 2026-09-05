#!/usr/bin/env python3
"""Audit equivalent-system full videos using complete latents and ordered routes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.offline_eval import output_error_metrics


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for part in iter(lambda:handle.read(1024*1024),b''):
            digest.update(part)
    return digest.hexdigest()


def load_case(root):
    paths=list(Path(root).rglob('case_state.json'))
    if len(paths)!=1:
        raise ValueError(f'exactly one case required, found {len(paths)} in {root}')
    state=json.loads(paths[0].read_text())
    case=paths[0].parent
    stats=json.loads((case/'sparse_history_stats.json').read_text())
    config=json.loads((case/'case_config.json').read_text())
    if config['case_key']!=state['case_key']:
        raise ValueError('config/state identity mismatch')
    identity_sha=hashlib.sha256(json.dumps(state['case_key'],ensure_ascii=False,separators=(',',':'),sort_keys=True).encode()).hexdigest()
    if identity_sha!=state['case_key_sha256']:
        raise ValueError('case identity digest mismatch')
    if state['status']!='pass' or any(state.get(key,0) for key in ('failed_calls','fallback_calls','nan_calls')):
        raise ValueError('technical pass without fallback is required')
    if state.get('complete_attention_capture') or state.get('timing_scope','').startswith('capture'):
        raise ValueError('capture-augmented wall time is not eligible for speed comparison')
    latent=torch.load(case/'latents.pt',map_location='cpu',weights_only=True)
    if not torch.isfinite(latent).all():
        raise ValueError('nonfinite latents')
    if latent.shape[1]!=config['latent_frames'] or state['decoded_frames']!=4*config['latent_frames']-3:
        raise ValueError('latent/pixel length mismatch')
    if sha(case/'video.mp4')!=state['video_sha256']:
        raise ValueError('video hash does not match terminal state')
    records=stats.get('call_records',[])
    if not records or len(records)!=stats['calls']:
        raise ValueError('complete ordered call records required')
    ordered=[(row['layer_id'],row['current_start'],row['denoising_pass'],row['route_plan_sha256']) for row in records]
    return case,state,stats,config,latent,ordered


def compare(reference_root,candidate_root):
    ref,cur=load_case(reference_root),load_case(candidate_root)
    for key in ('method','prompt_id','prompt','seed','latent_frames','history_density',
                'rope_policy','refresh_policy','backend','method_params'):
        if ref[3].get(key)!=cur[3].get(key):
            raise ValueError(f'not a same-method comparison: {key}')
    error=output_error_metrics(ref[4],cur[4])
    same_route=ref[5]==cur[5]
    numerical=error['max_abs']<=.02 and error['relative_l2']<=.01 and error['one_minus_cosine']<=.001
    wall_ref,wall_cur=ref[1]['end_to_end_s'],cur[1]['end_to_end_s']
    rows=[]
    for case,state,stats,config,latent,ordered in (ref,cur):
        records=stats['call_records']
        rows.append({'case':str(case),'case_identity_sha256':state['case_key_sha256'],
            'source_commit':config.get('execution_commit',config['case_key']['commit']),
            'end_to_end_s':state['end_to_end_s'], 'model_load_s_total':state.get('model_load_s_total'),
            'end_to_end_with_amortized_load_s':state.get('end_to_end_with_amortized_load_s'),
            'peak_allocated_gb':state.get('peak_allocated_gb'), 'timing':stats['timing'],
            'materialize_total_s':sum(row.get('materialize_total_s',0) for row in records),
            'backend_complete_s':sum(row.get('backend_complete_s',0) for row in records),
            'transferred_bytes':stats['transferred_bytes'],'history_union_cache':stats.get('history_union_cache'),
            'ordered_route_sha256':hashlib.sha256(json.dumps(ordered).encode()).hexdigest(),
            'latents_sha256':sha(case/'latents.pt'),'video_sha256':sha(case/'video.mp4'),
            'state_sha256':sha(case/'case_state.json'),'stats_sha256':sha(case/'sparse_history_stats.json')})
    return {'status':'pass' if same_route and numerical else 'fail','same_ordered_routes':same_route,
        'bitwise_equal_latents':bool(torch.equal(ref[4],cur[4])),'latent_error':error,
        'end_to_end_speedup':wall_ref/wall_cur,'end_to_end_reduction':1-wall_cur/wall_ref,
        'rows':rows,'timing_scope':'generation + VAE + finite checks + artifact save/encode/decode audit; model load separate',
        'replicates':1,'absolute_quality_review_pending':True,
        'general_speedup_claim':False,'hardware_counter_claim':False,
        'warning':'Component service times are not added to infer overlap or critical-path savings.'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',required=True)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    result=compare(args.reference,args.candidate)
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='rows'},indent=2))
    if result['status']!='pass':
        raise SystemExit(1)


if __name__=='__main__':
    main()
