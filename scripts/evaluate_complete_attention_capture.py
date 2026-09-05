#!/usr/bin/env python3
"""Offline full-Attention audit with causal proxies and actual-byte controls.

All candidate routes are constructed before the Dense teacher is evaluated.
Historical raw K is used to reconstruct archive prototypes, never exposed as an
argument to the online utility selector. No formal promotion occurs here.
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
from adapters.longlive_sparse.ar_routing import build_route_plan
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.offline_eval import dense_history_attention, routed_history_attention, output_error_metrics
from adapters.longlive_sparse.route_plan import HistoryRoutePlan, map_union_coordinates
from adapters.longlive_sparse.selectors import summarize_query_for_pretransfer
from adapters.longlive_sparse.system_utility_route import build_system_utility_route, SystemUtilityRouteConfig
from adapters.longlive_sparse.utility import apply_query_group_policy


def validate_capture(capture):
    required={'query','query_unrotated','key','key_unrotated','value','exact_key','exact_value',
              'frame_ids','token_ids','spatial_height','spatial_width','route_plan'}
    if not required.issubset(capture) or not capture.get('contains_sink_current_recent'):
        raise ValueError('complete post-RoPE capture with raw proxy inputs is required')
    if capture.get('teacher_used_by_selector') is not False or not capture.get('scope','').endswith('post_rope'):
        raise ValueError('invalid offline teacher boundary')
    if capture['query'].shape[0]!=1:
        raise ValueError('this development evaluator is explicitly batch-one')
    for raw,rotated in (('query_unrotated','query'),('key_unrotated','key')):
        if capture[raw].shape!=capture[rotated].shape:
            raise ValueError('raw/post-RoPE geometry mismatch')


def matched_legacy_route(archive, summary, frame_ids, token_ids, candidate_frames, candidate):
    """Rerun the actual legacy selector at each candidate head's used budget.

    This is not a coordinate-prefix truncation or teacher-ranked control.
    It matches exact compact payload AND rectangular padding. Legacy retains
    its token-trimmed tier boundary; the granularity difference remains explicit.
    """
    counts=(candidate.union_frame_ids>=0).sum(-1)[0]
    selections=[]
    for h, count in enumerate(counts.tolist()):
        if count<1:
            raise ValueError('empty utility union cannot be matched to legacy minimum-one admission')
        local=HistoryArchive(replace(archive.config,history_density=count/candidate.candidate_history_tokens),
            spatial_height=archive.spatial_height,spatial_width=archive.spatial_width)
        local._layers[0]={frame_id:replace(frame,key=frame.key[:,:,h:h+1],value=frame.value[:,:,h:h+1],
            block_centroids=frame.block_centroids[:,h:h+1],block_value_centroids=frame.block_value_centroids[:,h:h+1])
            for frame_id,frame in archive._layers[0].items()}
        local_summary=replace(summary,query_labels=summary.query_labels[:,h:h+1],
            query_centroids=summary.query_centroids[:,h:h+1],query_group_sizes=summary.query_group_sizes[:,h:h+1])
        route=local.route_indexed(0,local_summary,candidate_frames,exact_k_tokens=candidate.exact_k_tokens)
        assert route.unique_history_tokens==count
        mapped=map_union_coordinates(route,frame_ids[:,h:h+1],token_ids[:,h:h+1])
        selections.append([mapped[0,0,:count]])
    route=build_route_plan(method='legacy_final_matched_byte_control',routing_stage='pre-transfer',
        query_labels=torch.zeros_like(candidate.query_labels),selections=[selections],
        history_frame_ids=frame_ids,history_token_ids=token_ids,
        candidate_history_tokens=candidate.candidate_history_tokens,exact_k_tokens=candidate.exact_k_tokens,
        density=candidate.target_history_density,metadata={'matched_to_route':candidate.digest(),
            'actual_tokens_per_head':counts.tolist(),'teacher_used':False,'legacy_tiers_rerun':True})
    assert torch.equal((route.union_frame_ids>=0).sum(-1),(candidate.union_frame_ids>=0).sum(-1))
    assert route.union_frame_ids.shape==candidate.union_frame_ids.shape
    return route


def construct_routes(capture, *, device='cpu'):
    validate_capture(capture)
    params=json.loads((ROOT/'configs/formal/method_params.json').read_text())['method_params']['transfer_vaware_hybrid_history']
    config=SparseHistoryConfig(method='transfer_vaware_hybrid_history',history_density=.25,method_params=params)
    archive=HistoryArchive(config,spatial_height=capture['spatial_height'],spatial_width=capture['spatial_width'])
    frame_ids,token_ids=capture['frame_ids'].long().cpu(),capture['token_ids'].long().cpu()
    frames=list(dict.fromkeys(frame_ids[0,0].tolist()))
    for frame_id in frames:
        mask=frame_ids[0,0]==frame_id
        order=token_ids[0,0,mask].argsort()
        k=capture['key_unrotated'][:,mask][:,order].cpu()
        v=capture['value'][:,mask][:,order].cpu()
        archive.index_frame(0,frame_id,k.to(device),v,storage_k=k,storage_v=v)
    summary=summarize_query_for_pretransfer(capture['query_unrotated'].to(device),64)
    context=archive.online_routing_context(0,summary,frames)
    legacy=archive.route_indexed(0,summary,frames,exact_k_tokens=capture['exact_key'].shape[1])
    routes={'captured_route':HistoryRoutePlan.from_state_dict(capture['route_plan']), 'legacy_cap25':legacy,
            'legacy_top_p095':apply_query_group_policy(legacy,context,policy='mass_preserving_top_p',top_p=.95,min_k_ratio=.1)}
    for value in ('peak_value','count_uniform'):
        candidate=build_system_utility_route(context,exact_k_tokens=capture['exact_key'].shape[1],
            config=SystemUtilityRouteConfig(value_candidate=value,cost_strategy='static_block'))
        routes[value]=candidate
        routes['legacy_matched_'+value]=matched_legacy_route(archive,summary,frame_ids,token_ids,frames,candidate)
    return routes


def retained_probability_mass(capture, routes, *, device='cpu', chunk_size=64):
    """Exact offline teacher mass, including the non-sparse exact KV context."""
    q,k,ek=(capture[name].to(device) for name in ('query','key','exact_key'))
    maps={name:map_union_coordinates(route,capture['frame_ids'],capture['token_ids'])
          for name,route in routes.items()}
    all_mass={name:[] for name in routes}
    all_history={name:[] for name in routes}
    history_share=[]
    for b in range(q.shape[0]):
        for h in range(q.shape[2]):
            full_key=torch.cat((ek[b,:,h],k[b,:,h])).float()
            masks={}
            for name,route in routes.items():
                groups=route.query_group_sizes.shape[-1]
                mask=torch.zeros(groups,k.shape[1],dtype=torch.bool)
                for group in range(groups):
                    count=int(route.group_history_counts[b,h,group])
                    indices=route.group_union_indices[b,h,group,:count].cpu()
                    mask[group,maps[name][b,h].cpu().index_select(0,indices)]=True
                masks[name]=mask.to(device)
            for start in range(0,q.shape[1],chunk_size):
                end=min(start+chunk_size,q.shape[1])
                p=torch.softmax(q[b,start:end,h].float()@full_key.T*q.shape[-1]**-.5,dim=-1)
                exact_mass=p[:,:ek.shape[1]].sum(-1)
                history=p[:,ek.shape[1]:]
                denominator=history.sum(-1)
                history_share.append(denominator.cpu())
                for name,route in routes.items():
                    labels=route.query_labels[b,h,start:end].long().to(device)
                    selected=(history*masks[name].index_select(0,labels)).sum(-1)
                    all_mass[name].append((exact_mass+selected).cpu())
                    all_history[name].append((selected/denominator.clamp_min(1e-20)).cpu())
    result={}
    for name in routes:
        total=torch.cat(all_mass[name])
        historical=torch.cat(all_history[name])
        result[name]={'retained_total_mass_mean':float(total.mean()),
            'retained_total_mass_min':float(total.min()),'retained_total_mass_p05':float(total.quantile(.05)),
            'retained_history_mass_mean':float(historical.mean()),
            'retained_history_mass_p05':float(historical.quantile(.05)),
            'full_teacher_history_mass_mean':float(torch.cat(history_share).mean())}
    return result


@torch.inference_mode()
def evaluate(capture, *, device='cpu'):
    # Complete the route construction stage before creating any teacher output.
    routes=construct_routes(capture,device=device)
    q,k,v,ek,ev=(capture[name].to(device) for name in ('query','key','value','exact_key','exact_value'))
    teacher=dense_history_attention(q,torch.cat((ek,k),1),torch.cat((ev,v),1))
    masses=retained_probability_mass(capture,routes,device=device)
    records={}
    bytes_per_token=2*q.shape[-1]*q.element_size()
    for name,route in routes.items():
        candidate=routed_history_attention(q,k,v,capture['frame_ids'],capture['token_ids'],route,
                                            exact_key=ek,exact_value=ev)
        records[name]={'output_error':output_error_metrics(teacher,candidate),'route':route.as_dict(),
            'probability_mass':masses[name],
            'actual_tokens_per_head':(route.union_frame_ids>=0).sum(-1).tolist(),
            'unique_payload_bytes':route.unique_history_tokens*bytes_per_token,
            'padded_compact_bytes':route.union_frame_ids.numel()*bytes_per_token,
            'executor_storage_estimate':route.grouped_executor_storage(head_dim=q.shape[-1],element_size=q.element_size())}
    return {'status':'pass','scope':'offline_complete_post_rope_attention','records':records,
            'exact_tokens':ek.shape[1],'historical_tokens':k.shape[1],
            'teacher_used_by_online_selector':False,'all_routes_built_before_teacher':True,
            'proxy_reconstruction_device':str(device),'formal_promotion_allowed':False,
            'budget_control':'rerun legacy 70/15/15 per head at utility actual count, not prefix truncation',
            'granularity_limit':'utility uses whole blocks; legacy retains token-trimmed tier boundaries'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args()
    torch.set_num_threads(2)
    capture=torch.load(args.capture,map_location='cpu',weights_only=True)
    result=evaluate(capture,device=args.device)
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    result['capture_sha256']=hashlib.sha256(Path(args.capture).read_bytes()).hexdigest()
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'cases':len(result['records']),
        'errors':{key:value['output_error'] for key,value in result['records'].items()}},indent=2))


if __name__=='__main__':
    main()
