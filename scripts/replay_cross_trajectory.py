#!/usr/bin/env python3
"""GPU-only full-context counterfactual evaluation of already-frozen routes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.offline_eval import dense_history_attention,routed_history_attention,output_error_metrics
from adapters.longlive_sparse.route_plan import HistoryRoutePlan


def evaluate(captures, plans, *, actor, device='cuda'):
    first=captures[0]
    rows=[]
    for call,capture in enumerate(captures):
        if (capture['layer'],capture['current_start'],capture['denoising_pass'])!=(first['layer'],first['current_start'],call):
            raise ValueError('mixed or incomplete capture sequence')
        for field in ('key','key_unrotated','value','frame_ids','token_ids','history_positions'):
            if not torch.equal(first[field],capture[field]):raise ValueError('history changed within fixed chunk')
        if capture['route_sha']!=first['route_sha']:raise ValueError('executed actor route was not per-chunk fixed')
        if actor in plans and plans[actor].digest()!=capture['route_sha']:raise ValueError('frozen own route mismatch')
        q,k,v,ek,ev=[capture[name].to(device) for name in ('query','key','value','exact_key','exact_value')]
        teacher=dense_history_attention(q,torch.cat((ek,k),1),torch.cat((ev,v),1))
        errors={}
        for method,plan in plans.items():
            output=routed_history_attention(q,k,v,capture['frame_ids'],capture['token_ids'],plan,exact_key=ek,exact_value=ev)
            errors[method]={'output_error':output_error_metrics(teacher,output),
                            'selected_tokens':plan.unique_history_tokens,'route_sha':plan.digest()}
        rows.append({'call':call,'phase':'clean_commit' if call==4 else 'denoising','methods':errors})
    if len(rows)!=5:raise ValueError('five calls required')
    return {'status':'pass','actor':actor,'layer':first['layer'],'current_start':first['current_start'],
        'teacher_scope':'full_retrieved_context_on_this_actor_trajectory','routes_frozen_before_teacher':True,
        'online_router_in_worker':False,'rows':rows}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-root',required=True)
    parser.add_argument('--task-file',required=True)
    parser.add_argument('--plan-dir',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('real CUDA required')
    root=Path(args.input_root)
    task=json.loads(Path(args.task_file).read_text())
    path=Path(args.plan_dir)/task['plan_file']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=task['plan_file_sha256']:raise ValueError('plan SHA mismatch')
    plans={name:HistoryRoutePlan.from_state_dict(state) for name,state in torch.load(path,map_location='cpu',weights_only=True)['plans'].items()}
    captures=[]
    for name,sha in zip(task['capture_files'],task['capture_sha256']):
        p=root/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError('capture SHA mismatch')
        captures.append(torch.load(p,map_location='cpu',weights_only=True))
    start=time.perf_counter()
    result=evaluate(captures,plans,actor=task['actor'])
    result.update(task_id=task['task_id'],prompt=task['prompt'],gpu=torch.cuda.get_device_name(),
                  diagnostic_wall_s=time.perf_counter()-start)
    out=Path(args.output)
    with out.open('x') as handle:json.dump(result,handle,indent=2)
    print(json.dumps({'task':task['task_id'],'status':result['status'],'gpu':result['gpu']}),flush=True)


if __name__=='__main__':main()
