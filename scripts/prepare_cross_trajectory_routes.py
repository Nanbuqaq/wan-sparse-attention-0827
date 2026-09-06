#!/usr/bin/env python3
"""Freeze both causal routes on original-device captured inputs before H replay."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.route_plan import HistoryRoutePlan
from adapters.longlive_sparse.selectors import PretransferQuerySummary,summarize_query_for_pretransfer
from scripts.evaluate_complete_attention_capture import validate_capture

METHODS=('transfer_vaware_hybrid_history','rope_aligned_final_history')


def construct(capture, *, device):
    validate_capture(capture)
    if capture['denoising_pass']!=0 or capture['schema_version']<3:
        raise ValueError('first-call repaired capture required')
    source=capture['sparse_config']
    actor=source['method']
    if actor not in (*METHODS,'rag_dense') or source['rope_policy']!='upstream_zero':
        raise ValueError('unsupported actor/position policy')
    params=json.loads((ROOT/'configs/formal/method_params.json').read_text())['method_params'][METHODS[0]]
    frames=list(dict.fromkeys(capture['frame_ids'][0,0].tolist()))
    actual=capture.get('actual_online_context')
    plans,checks={},[]
    for method,space,q_name in ((METHODS[0],'unrotated','query_unrotated'),(METHODS[1],'post_rope','query')):
        cfg=SparseHistoryConfig(method=method,history_density=.25,refresh_policy='per_chunk',method_params=params)
        archive=HistoryArchive(cfg,spatial_height=capture['spatial_height'],spatial_width=capture['spatial_width'])
        for frame in frames:
            mask=capture['frame_ids'][0,0]==frame
            order=capture['token_ids'][0,0,mask].argsort()
            k=capture['key_unrotated'][:,mask][:,order].cpu()
            v=capture['value'][:,mask][:,order].cpu()
            index=archive.index_frame(0,frame,k.to(device),v,storage_k=k,storage_v=v)
            if method==actor:
                block=actual['block_frame_ids']==frame
                errors={name:float((actual[field][:,:,block]-getattr(index,name)).abs().max())
                    for name,field in [('block_centroids','key_prototypes'),('block_value_centroids','value_prototypes')]}
                if max(errors.values())>1e-5:
                    raise ValueError(f'original-device committed prototype reconstruction failed: {errors}')
                checks.append({'frame':frame,'errors':errors})
                archive._layers[0][frame]=replace(index,block_centroids=actual['key_prototypes'][:,:,block],
                                                  block_value_centroids=actual['value_prototypes'][:,:,block])
        summary=(PretransferQuerySummary(**capture['actual_query_summary']) if method==actor else
                 summarize_query_for_pretransfer(capture[q_name].to(device),64,coordinate_space=space))
        if summary.coordinate_space!=space:
            raise ValueError('actual Q summary coordinate-space mismatch')
        plans[method]=archive.route_indexed(0,summary,frames,exact_k_tokens=capture['exact_key'].shape[1])
    captured=HistoryRoutePlan.from_state_dict(capture['route_plan'])
    if actor in METHODS and plans[actor].digest()!=captured.digest():
        raise ValueError('own first-route reproduction failed')
    if actor=='rag_dense' and captured.history_pair_density!=1.:
        raise ValueError('Dense source did not execute complete retrieved history')
    counts=[(p.union_frame_ids>=0).sum(-1) for p in plans.values()]
    if not torch.equal(*counts):
        raise ValueError('counterfactual budgets differ by head')
    return plans,checks


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture-root',required=True)
    parser.add_argument('--kind',choices=['motion','state'],required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('use original physical CUDA lane for frozen proxies')
    root=Path(args.capture_root).resolve()
    audit=json.loads((root/'trajectory_audit.json').read_text())
    if audit['status']!='pass':raise ValueError('exact original-trajectory gate required')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    tasks=[]
    for row in audit['cases']:
        if row['prompt']!=f'calibration_{args.kind}':continue
        directory=Path(row['capture_dir'])
        for layer in (0,9,19,29):
            for start in (28080,46800):
                path=directory/f'layer{layer:02d}_start{start:08d}_pass00.pt'
                payload=torch.load(path,map_location='cpu',weights_only=True)
                plans,checks=construct(payload,device='cuda')
                task_id=f'{row["method"]}__{row["prompt"]}__L{layer:02d}__start{start}'
                plan_path=out/f'{task_id}.pt'
                torch.save({'plans':{name:plan.state_dict() for name,plan in plans.items()},
                    'actor':row['method'],'case_id':row['id'],'prototype_checks':checks},plan_path)
                paths=[directory/f'layer{layer:02d}_start{start:08d}_pass{call:02d}.pt' for call in range(5)]
                tasks.append({'task_id':task_id,'actor':row['method'],'prompt':row['prompt'],'layer':layer,'start':start,
                    'plan_file':plan_path.name,'plan_file_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                    'capture_files':[str(p.relative_to(root)) for p in paths],
                    'capture_sha256':[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
                    'generation_commit':audit['generation_commit']})
                print(json.dumps({'prepared':task_id,'own_route_reproduced':True}),flush=True)
    if len(tasks)!=24:raise ValueError('expected three actors x four layers x two history points')
    (out/'tasks.json').write_text(json.dumps({'status':'pass','kind':args.kind,'gpu':torch.cuda.get_device_name(),
        'scope':'original_device_frozen_routes_before_teacher','tasks':tasks},indent=2)+'\n')


if __name__=='__main__':main()
