#!/usr/bin/env python3
"""Full FP32 teacher comparison of executed/fresh/two-phase diagnostic routes.

All choices are current or earlier-call causal proxies on ONE baseline path;
these counterfactual Attention errors do not replace on-policy video testing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.offline_eval import dense_history_attention, routed_history_attention, output_error_metrics
from adapters.longlive_sparse.route_plan import HistoryRoutePlan
from scripts.evaluate_complete_attention_capture import construct_routes, retained_probability_mass


def evaluate(paths, *, device):
    first, late = None, None
    records = []
    expected = None
    for path in sorted(paths):
        payload=torch.load(path,map_location='cpu',weights_only=True)
        marker=(payload['layer'],payload['current_start'])
        if expected is None:
            expected=marker
        if marker != expected:
            raise ValueError('do not mix layers/chunks/cases in a denoising sequence')
        call=int(payload['denoising_pass'])
        if call != len(records):
            raise ValueError('complete ordered five-call sequence required')
        constructed=construct_routes(payload,device=device,value_candidates=())
        fresh=constructed['legacy_cap25']
        executed=HistoryRoutePlan.from_state_dict(payload['route_plan'])
        if first is None:
            first=fresh
            if executed.digest()!=fresh.digest():
                raise RuntimeError('first-pass actual online input did not reproduce executed route')
        if call == 2:
            late=fresh
        two_phase=first if call<2 else late
        routes={'executed_per_chunk':executed,'fresh_per_call':fresh,'early_late_2plan':two_phase}
        q,k,v,ek,ev=[payload[name].to(device) for name in ('query','key','value','exact_key','exact_value')]
        teacher=dense_history_attention(q,torch.cat((ek,k),1),torch.cat((ev,v),1))
        outputs={name:routed_history_attention(q,k,v,payload['frame_ids'],payload['token_ids'],plan,
                   exact_key=ek,exact_value=ev) for name,plan in routes.items()}
        mass=retained_probability_mass(payload,routes,device=device)
        records.append({'call':call,'phase':'clean_context_commit' if call==4 else 'denoising',
            'capture_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'query_summary_source':payload.get('query_summary_source'),
            'records':{name:{'route_sha':plan.digest(),'selected_tokens':plan.unique_history_tokens,
                'vs_full_teacher':output_error_metrics(teacher,outputs[name]),
                'vs_executed_route':output_error_metrics(outputs['executed_per_chunk'],outputs[name]),
                **mass[name]} for name,plan in routes.items()}})
        print(json.dumps({'stage':'evaluated','layer':marker[0],'start':marker[1],'call':call,
              'teacher_relative_l2':{name:row['vs_full_teacher']['relative_l2']
                                     for name,row in records[-1]['records'].items()}}),flush=True)
    if len(records)!=5:
        raise ValueError('four denoising calls plus clean-context commit are required')
    return {'status':'pass','layer':expected[0],'current_start':expected[1],
            'scope':'offline_counterfactual_on_frozen_baseline_trajectory','records':records,
            'new_routes_used_by_video':False,'on_policy_quality_claim':False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture-dir',required=True)
    parser.add_argument('--layer',type=int,required=True)
    parser.add_argument('--start',type=int,required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    torch.set_num_threads(2)
    paths=list(Path(args.capture_dir).glob(f'layer{args.layer:02d}_start{args.start:08d}_pass*.pt'))
    out=Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    result=evaluate(paths,device=args.device)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
