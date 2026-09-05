#!/usr/bin/env python3
"""Bounded capture experiment for an index-time, execution-aligned Q/K proxy."""
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
from scripts.evaluate_complete_attention_capture import validate_capture, retained_probability_mass
from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.phase_prototypes import canonical_wan_frequency_table, archive_rope0_key
from adapters.longlive_sparse.selectors import summarize_query_for_pretransfer
from adapters.longlive_sparse.offline_eval import dense_history_attention, routed_history_attention, output_error_metrics
from adapters.longlive_sparse.route_plan import HistoryRoutePlan
from adapters.longlive_sparse.system_utility_route import SystemUtilityRouteConfig,build_system_utility_route


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():
        raise RuntimeError('real CUDA required')
    capture=torch.load(args.capture,map_location='cpu',weights_only=True)
    validate_capture(capture)
    if capture.get('sparse_config', {}).get('method', 'transfer_vaware_hybrid_history') != 'transfer_vaware_hybrid_history':
        raise ValueError('this pilot uses a frozen legacy-Final trajectory, not an already aligned method')
    if capture.get('rope_policy')!='upstream_zero' or bool(capture['history_positions'][...,0].any()):
        raise ValueError('only the proved fixed historical RoPE policy is eligible')
    params=json.loads((ROOT/'configs/formal/method_params.json').read_text())['method_params']['transfer_vaware_hybrid_history']
    archive=HistoryArchive(SparseHistoryConfig(method='transfer_vaware_hybrid_history',history_density=.25,method_params=params),
        spatial_height=capture['spatial_height'],spatial_width=capture['spatial_width'])
    frames=list(dict.fromkeys(capture['frame_ids'][0,0].tolist()))
    freqs=canonical_wan_frequency_table(capture['query'].shape[-1]).cuda()
    reconstruction_error=0.
    for frame in frames:
        mask=capture['frame_ids'][0,0]==frame
        order=capture['token_ids'][0,0,mask].argsort()
        raw=capture['key_unrotated'][:,mask][:,order]
        value=capture['value'][:,mask][:,order]
        # Reconstruct from RAW archived K and fixed coordinates; do not use the
        # captured post-RoPE K to build the selector prototype.
        rotated=archive_rope0_key(raw.cuda(),spatial_height=archive.spatial_height,
                                  spatial_width=archive.spatial_width,freqs=freqs)
        expected=capture['key'][:,mask][:,order].cuda()
        delta=float((rotated.float()-expected.float()).abs().max())
        reconstruction_error=max(reconstruction_error,delta)
        if delta!=0.:
            raise ValueError(f'index-time rotation does not reconstruct actual executed K: {delta}')
        archive.index_frame(0,frame,rotated,value,storage_k=raw,storage_v=value)
    summary=summarize_query_for_pretransfer(capture['query'].cuda(),64)
    context=archive.online_routing_context(0,summary,frames)
    exact=capture['exact_key'].shape[1]
    routes={'captured_legacy':HistoryRoutePlan.from_state_dict(capture['route_plan']),
            'rope0_final':replace(archive.route_indexed(0,summary,frames,exact_k_tokens=exact),method='rope0_final_capture_candidate')}
    for value in ('peak_value','count_uniform'):
        routes['rope0_'+value]=replace(build_system_utility_route(context,exact_k_tokens=exact,
            config=SystemUtilityRouteConfig(value_candidate=value,cost_strategy='static_block')),
            method='rope0_'+value+'_capture_candidate')
    # Teacher outputs and weights are computed only after all routes are fixed.
    q,k,v,ek,ev=(capture[name].cuda() for name in ('query','key','value','exact_key','exact_value'))
    teacher=dense_history_attention(q,torch.cat((ek,k),1),torch.cat((ev,v),1))
    masses=retained_probability_mass(capture,routes,device='cuda')
    records={}
    for name,route in routes.items():
        output=routed_history_attention(q,k,v,capture['frame_ids'],capture['token_ids'],route,exact_key=ek,exact_value=ev)
        records[name]={'error':output_error_metrics(teacher,output),'probability_mass':masses[name],
                      'route':route.as_dict(),'actual_tokens_per_head':(route.union_frame_ids>=0).sum(-1).tolist()}
    result={'status':'pass','scope':'capture_hypothesis_only','runtime_integrated':False,
        'formal_promotion_allowed':False,'index_time_key_reconstruction_max_abs':reconstruction_error,
        'selector_post_rope_K_from_raw_archive_not_teacher':True,'teacher_output_used_by_selector':False,
        'records':records,'capture_sha256':hashlib.sha256(Path(args.capture).read_bytes()).hexdigest()}
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({name:record['error'] for name,record in records.items()},indent=2))


if __name__=='__main__':
    main()
