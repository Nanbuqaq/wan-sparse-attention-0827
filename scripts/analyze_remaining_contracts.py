#!/usr/bin/env python3
"""Causal prefetch screen, matched-byte budget ablations, censored fine revisit."""
import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.contexts import OnlineRoutingContext
from adapters.longlive_sparse.route_plan import HistoryRoutePlan
from adapters.longlive_sparse.system_utility_route import SystemUtilityRouteConfig, build_system_utility_route
from adapters.longlive_sparse.offline_eval import dense_history_attention, routed_history_attention, output_error_metrics
from scripts.probe_verified_prefetch_routes import blocks, width, admit
from scripts.audit_candidate_permutation import reconstruct
from scripts.evaluate_complete_attention_capture import matched_legacy_route, retained_probability_mass


def exact_tokens(plan):
    return {(h,int(f),int(t)) for h in range(plan.union_frame_ids.shape[1])
            for f,t in zip(plan.union_frame_ids[0,h].tolist(),plan.union_token_ids[0,h].tolist()) if f>=0}


def q_to_next_prediction(context, causal_candidates):
    """Accepts only source Q summary + previously indexed target prototypes."""
    q,k = context.query_centroids.float(),context.key_prototypes.float()
    widths = (context.block_token_ends-context.block_token_starts).float()
    logits = torch.einsum('bhgd,bhnd->bhgn',q,k)/math.sqrt(q.shape[-1])+widths.log()
    groups = context.query_group_sizes.float()
    importance = (logits.softmax(-1)*(groups/groups.sum(-1,keepdim=True)).unsqueeze(-1)).sum(2)
    ordered = []
    for h in range(q.shape[1]):
        for i in torch.argsort(importance[0,h]/widths,descending=True,stable=True).tolist():
            ordered.append((h,int(context.block_frame_ids[i]),int(context.block_token_starts[i])//64))
    return admit(ordered,causal_candidates,q.shape[1])


def traffic(prediction, actual):
    predicted = {(h,f,t) for h,f,b in prediction for t in range(b*64,b*64+width((h,f,b)))}
    hit,missing,extra = actual & predicted,actual-predicted,predicted-actual
    return {'predicted_bytes':len(predicted)*512,'actual_exact_bytes':len(actual)*512,
        'hit_bytes':len(hit)*512,'miss_bytes':len(missing)*512,'extra_bytes':len(extra)*512,
        'recall':len(hit)/len(actual),'precision':len(hit)/len(predicted) if predicted else None,
        'prefetch_plus_exact_completion_bytes':(len(predicted)+len(missing))*512,
        'traffic_ratio_vs_exact_no_prefetch':(len(predicted)+len(missing))/len(actual),
        'final_required_tokens_exact':(hit|missing)==actual,
        'timeliness':None,'measured_exposed_wait':None,'GPU_prefetch_executed':False}


def prefetch_screen(root):
    records = torch.load(root/'contexts.pt',map_location='cpu',weights_only=True)['records']
    lookup = {(r['source_layer'],r['current_start']):r for r in records}
    positions = {key:i for i,key in enumerate(lookup)}
    rows=[]
    for record in records:
        layer,target,start = record['source_layer'],record['target_layer'],record['current_start']
        if (target,start) not in lookup:
            continue
        if not record['target_layer_not_yet_executed'] or record['target_Q_or_route_used'] or positions[layer,start]>=positions[target,start]:
            raise ValueError('invalid source-before-target causal boundary')
        causal = record['source_candidate_frame_ids']
        available = record['target_frames_already_indexed']
        source_plan=HistoryRoutePlan.from_state_dict(record['source_route'])
        heads = source_plan.union_frame_ids.shape[1]
        predictors = {'q_to_next_proto':set()}
        if record['next_context'] is not None:
            predictors['q_to_next_proto']=q_to_next_prediction(OnlineRoutingContext(**record['next_context']),causal)
        previous_layer = [b for b in sorted(blocks(source_plan)) if b[1] in available]
        predictors['previous_adjacent_route']=admit(previous_layer,causal,heads)
        if (target,start-4680) in lookup:
            past=HistoryRoutePlan.from_state_dict(lookup[target,start-4680]['source_route'])
            predictors['previous_chunk_route']=admit([b for b in sorted(blocks(past)) if b[1] in available],causal,heads)
        universe=[(h,f,b) for h in range(heads) for f in available for b in range(25)]
        random.Random(20260904+start+target*701).shuffle(universe)
        predictors['random_available_control']=admit(universe,causal,heads)
        # Actual target route/summary is first read AFTER every prediction.
        target_record=lookup[target,start]
        actual_plan=HistoryRoutePlan.from_state_dict(target_record['source_route'])
        actual=exact_tokens(actual_plan)
        for name,predicted in predictors.items():
            rows.append({'source_layer':layer,'target_layer':target,'current_start':start,'predictor':name,
                'causal_candidate_frames':causal,'target_frames_already_indexed':available,
                'target_frames_unavailable_at_prediction':record['target_missing_at_prediction'],
                'target_candidate_roster_matches_source':target_record['source_candidate_frame_ids']==causal,
                'same_target_logical_route':actual_plan.digest(),'all_KV_owned_by_target_layer':True,
                **traffic(predicted,actual)})
    grouped=defaultdict(list)
    for row in rows: grouped[row['predictor']].append(row)
    summary={}
    for name,items in grouped.items():
        actual=sum(r['actual_exact_bytes'] for r in items); predicted=sum(r['predicted_bytes'] for r in items)
        hit=sum(r['hit_bytes'] for r in items)
        summary[name]={'calls':len(items),'byte_recall':hit/actual,
            'byte_precision':hit/predicted if predicted else None,
            'extra_bytes':sum(r['extra_bytes'] for r in items),
            'traffic_ratio_vs_exact':sum(r['prefetch_plus_exact_completion_bytes'] for r in items)/actual}
    return {'status':'pass','records':rows,'summary':summary,
        'boundary':'after source attention, before target layer; only preexisting target prototypes',
        'raw_layer_head_spaces_and_RoPE_are_not_assumed_aligned':True,
        'screen_only_not_timing_or_video_speed_evidence':True}


@torch.inference_mode()
def teacher_blocks(capture,context,device):
    q,k,ek = [capture[n].to(device) for n in ('query','key','exact_key')]
    n=context.blocks
    scores=torch.zeros(q.shape[2],n,dtype=torch.float64)
    coordinate={(int(f),int(s)//64):i for i,(f,s) in enumerate(zip(context.block_frame_ids,context.block_token_starts))}
    for h in range(q.shape[2]):
        ids=torch.tensor([coordinate[int(f),int(t)//64] for f,t in zip(capture['frame_ids'][0,h],capture['token_ids'][0,h])],device=device)
        full=torch.cat((ek[0,:,h],k[0,:,h])).float()
        mass=torch.zeros(n,device=device,dtype=torch.float32)
        for start in range(0,q.shape[1],128):
            p=(q[0,start:start+128,h].float()@full.T/math.sqrt(q.shape[-1])).softmax(-1)
            mass.index_add_(0,ids,p[:,ek.shape[1]:].sum(0))
        scores[h]=mass.double().cpu()/q.shape[1]
    widths=(context.block_token_ends-context.block_token_starts).float()
    ranked=[]
    universe=set()
    for h in range(q.shape[2]):
        for i in torch.argsort(scores[h]/widths,descending=True,stable=True).tolist():
            b=(h,int(context.block_frame_ids[i]),int(context.block_token_starts[i])//64)
            ranked.append(b); universe.add(b)
    candidates=context.metadata['candidate_frame_ids']
    return admit(ranked,candidates,q.shape[2]),universe


@torch.inference_mode()
def evaluate_capture(path,protocol,device):
    capture=torch.load(path,map_location='cpu',weights_only=True)
    archive,summary,frames=reconstruct(capture)
    context=OnlineRoutingContext(**capture['actual_online_context'])
    baseline=archive.route_indexed(0,summary,frames,exact_k_tokens=capture['exact_key'].shape[1])
    if baseline.digest()!=capture['route_sha']:
        raise ValueError('captured causal inputs did not reproduce executed first-call route')
    routes={'legacy_cap25':baseline}
    configs={}
    for value in protocol['utility_candidates']:
        for name,fractions in protocol['budget_ablations'].items():
            cfg=SystemUtilityRouteConfig(value_candidate=value,cost_strategy='static_block',
                correlation_fraction=fractions[0],coverage_fraction=fractions[1],
                remote_fraction=fractions[2],exploration_fraction=fractions[3])
            key=f'{value}__{name}'
            route=build_system_utility_route(context,exact_k_tokens=baseline.exact_k_tokens,config=cfg)
            routes[key]=route; configs[key]=cfg.as_dict()
            routes['matched_legacy__'+key]=matched_legacy_route(archive,summary,capture['frame_ids'],capture['token_ids'],frames,route)
    # No full teacher data enters a route builder above.
    q,k,v,ek,ev=[capture[n].to(device) for n in ('query','key','value','exact_key','exact_value')]
    teacher=dense_history_attention(q,torch.cat((ek,k),1),torch.cat((ev,v),1))
    mass=retained_probability_mass(capture,routes,device=device)
    output={}
    for name,route in routes.items():
        actual=routed_history_attention(q,k,v,capture['frame_ids'],capture['token_ids'],route,exact_key=ek,exact_value=ev)
        output[name]={'route_sha':route.digest(),'output_error':output_error_metrics(teacher,actual),
            'tokens_per_head':(route.union_frame_ids>=0).sum(-1).tolist(),
            'payload_bytes':route.unique_history_tokens*512,'rectangular_bytes':route.union_frame_ids.numel()*512,
            'logical_density':route.history_pair_density,
            'scheduled_pairs':route.history_pairs+route.query_labels.numel()*route.exact_k_tokens,
            'probability_mass':mass[name]}
    top,universe=teacher_blocks(capture,context,device)
    return {'status':'pass','layer':capture['layer'],'current_start':capture['current_start'],
        'records':output,'configs':configs,'capture_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'all_routes_built_before_teacher':True,'exploration5_is_age_prior_fallback_not_random_or_bandit':True,
        'realized_bytes_differ_but_every_utility_has_exact_byte_matched_legacy_control':True,
        'scheduled_pair_metric_does_not_measure_internal_MMA_padding':True,
        'teacher_top_blocks':sorted(top),'candidate_blocks':sorted(universe),'executed_blocks':sorted(blocks(baseline)),
        'teacher_top_rank':'mean_full_attention_probability_per_history_token; whole-block25% physical cap'}


def feedback_gate(records):
    groups=defaultdict(dict)
    for r in records: groups[r['layer']][r['current_start']]=r
    rows=[]
    for layer,steps in sorted(groups.items()):
        initial=steps[46800]
        unseen=set(map(tuple,initial['candidate_blocks']))-set(map(tuple,initial['executed_blocks']))
        future=[steps[s] for s in (51480,56160)]
        available=set().union(*(set(map(tuple,r['candidate_blocks'])) for r in future))
        top=set().union(*(set(map(tuple,r['teacher_top_blocks'])) for r in future))
        hit=unseen & top; censored=unseen-available
        rows.append({'layer':layer,'initial_unselected_blocks':len(unseen),'future_top_reentries':len(hit),
            'not_observed_in_future_coarse_candidates':len(censored),
            'observed_lower_bound_reentry_fraction':len(hit)/len(unseen),
            'observable_subset_reentry_fraction':len(hit)/len(unseen & available) if unseen & available else None,
            'threshold10pct_observed':len(hit)/len(unseen)>=.1,
            'unobserved_is_not_a_safe_eviction_label':True})
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda');p.add_argument('--prefetch-only',action='store_true')
    p.add_argument('--gate-only',action='store_true',help='one real complete-capture branch gate, not the full analysis cohort')
    args=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    gate=json.loads((args.root/'summary.json').read_text())
    if gate['status']!='pass' or not gate['same_latents_routes_noise_and_future_prototypes']:
        raise ValueError('no-op generation/future-state gate required')
    args.output.mkdir(parents=True,exist_ok=False)
    prefetch=prefetch_screen(args.root)
    (args.output/'prefetch.json').write_text(json.dumps(prefetch,indent=2)+'\n')
    if args.prefetch_only:
        print(json.dumps(prefetch['summary'],indent=2));return
    protocol_path=ROOT/'configs/system/remaining_contracts_probe.json';protocol=json.loads(protocol_path.read_text())
    paths=sorted((args.root/'complete_attention_captures'/gate['prompt']).glob('*.pt'))
    if len(paths)!=6: raise ValueError('two layers and three consecutive chunks required')
    if args.gate_only:
        result=evaluate_capture(paths[0],protocol,args.device)
        (args.output/'gate.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'status':'pass','scope':'one_capture_GPU_branch_gate','capture':paths[0].name}))
        return
    records=[]
    for path in paths:
        result=evaluate_capture(path,protocol,args.device)
        (args.output/f'{path.stem}.json').write_text(json.dumps(result,indent=2)+'\n');records.append(result)
        print(json.dumps({'capture':path.name,'status':'pass'}),flush=True)
    result={'status':'pass','prompt':gate['prompt'],'records':records,'prefetch_summary':prefetch['summary'],
        'censored_feedback_gate':feedback_gate(records),'missing':0,'formal_or_video_promotion':False,
        'scope':'finite_development_counterfactuals_not_on_policy_quality',
        'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'analysis_source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'capture_source_commit':gate['source_commit'],
        'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest()}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()
