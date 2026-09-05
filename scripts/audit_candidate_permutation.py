#!/usr/bin/env python3
"""Separate same-set frame ordering from logical selection and numeric effects.

The legacy route is not changed. Reconstruct its actual captured causal inputs,
then permute only the order of the same candidate frames in isolated replay.
"""
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
from adapters.longlive_sparse.memory_dynamics import compare_coordinates
from adapters.longlive_sparse.route_plan import HistoryRoutePlan, map_union_coordinates
from adapters.longlive_sparse.selectors import PretransferQuerySummary, gather_per_head
from adapters.longlive_sparse.offline_eval import routed_history_attention, output_error_metrics
from scripts.evaluate_complete_attention_capture import validate_capture


def reconstruct(capture):
    validate_capture(capture)
    if capture.get('schema_version',0)<3 or capture.get('actual_online_context') is None:
        raise ValueError('repaired actual online context required')
    cfg=SparseHistoryConfig.from_mapping(capture['sparse_config'])
    if cfg.method!='transfer_vaware_hybrid_history':
        raise ValueError('this audit freezes legacy Final semantics')
    archive=HistoryArchive(cfg,spatial_height=capture['spatial_height'],spatial_width=capture['spatial_width'])
    ids=capture['frame_ids'][0,0]
    frames=list(dict.fromkeys(ids.tolist()))
    actual=capture['actual_online_context']
    for frame in frames:
        mask=ids==frame
        order=capture['token_ids'][0,0,mask].argsort()
        k=capture['key_unrotated'][:,mask][:,order].cpu()
        v=capture['value'][:,mask][:,order].cpu()
        archive.index_frame(0,frame,k,v,storage_k=k,storage_v=v)
        block_mask=actual['block_frame_ids']==frame
        stored=archive._layers[0][frame]
        error=float((actual['value_prototypes'][:,:,block_mask]-stored.block_value_centroids).abs().max())
        if error>1e-5:
            raise ValueError('saved V prototype inconsistent with committed V')
        archive._layers[0][frame]=replace(stored,block_centroids=actual['key_prototypes'][:,:,block_mask],
            block_value_centroids=actual['value_prototypes'][:,:,block_mask])
    summary=PretransferQuerySummary(**capture['actual_query_summary'])
    return archive,summary,frames


def audit(capture, *, attention_device=None):
    archive,summary,frames=reconstruct(capture)
    original=archive.route_indexed(0,summary,frames,exact_k_tokens=capture['exact_key'].shape[1])
    captured=HistoryRoutePlan.from_state_dict(capture['route_plan'])
    if original.digest()!=captured.digest():
        raise RuntimeError('original current summary does not reproduce captured first-pass route')
    orders={'original':frames,'sorted':sorted(frames),'reverse':frames[::-1],'rotate1':frames[1:]+frames[:1]}
    rows=[]
    original_fp32=original_bf16=None
    if attention_device:
        from adapters.longlive_sparse.backends import execute_plan
        q,k,v,ek,ev=[capture[name].to(attention_device) for name in ('query','key','value','exact_key','exact_value')]
    for name,order in orders.items():
        plan=archive.route_indexed(0,summary,order,exact_k_tokens=capture['exact_key'].shape[1])
        metric=compare_coordinates(original,plan,token_base=archive.spatial_height*archive.spatial_width)
        # Legacy Final at transfer_multiplier=1 gives each query the full union.
        full_union=lambda p: bool((p.group_history_counts == (p.union_frame_ids>=0).sum(-1).unsqueeze(-1)).all())
        same_edges=metric['jaccard']==1. and full_union(original) and full_union(plan)
        row={'order':name,'candidate_frames':order,'same_candidate_set':set(order)==set(frames),
             'route_sha':plan.digest(),'same_route_sha':plan.digest()==original.digest(),
             'coordinate_comparison':metric,'same_logical_edges':same_edges,
             'same_union_sequence':torch.equal(original.union_frame_ids,plan.union_frame_ids) and
                                   torch.equal(original.union_token_ids,plan.union_token_ids)}
        if attention_device:
            fp32=routed_history_attention(q,k,v,capture['frame_ids'],capture['token_ids'],plan,
                                           exact_key=ek,exact_value=ev)
            indices=map_union_coordinates(plan,capture['frame_ids'],capture['token_ids']).to(k.device)
            bf16=execute_plan('grouped_fa2',q,ek,ev,gather_per_head(k,indices),gather_per_head(v,indices),plan).output
            if original_fp32 is None:
                original_fp32,original_bf16=fp32,bf16
            row['fp32_vs_original_order']=output_error_metrics(original_fp32,fp32)
            row['bf16_fa2_vs_original_order']=output_error_metrics(original_bf16,bf16)
        rows.append(row)
    return {'status':'pass','scope':'same_candidate_set_order_audit_not_age_intervention',
            'layer':capture['layer'],'current_start':capture['current_start'],'rows':rows,
            'legacy_runtime_changed':False,'attention_device':attention_device}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--attention-device',choices=['cuda'])
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    path=Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    capture=torch.load(args.capture,map_location='cpu',weights_only=True)
    result=audit(capture,attention_device=args.attention_device)
    result['capture_sha256']=hashlib.sha256(Path(args.capture).read_bytes()).hexdigest()
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
