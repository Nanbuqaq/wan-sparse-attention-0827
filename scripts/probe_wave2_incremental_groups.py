#!/usr/bin/env python3
"""Real clean-arrival/access replay: Block64 versus bounded stable-ID K groups."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch


def replay(data,frame_tokens):
    groups=[];buckets={};token_group={};assignments=[];splits=[];evaluations=0;started=time.perf_counter()
    # Membership changes only on new committed tokens; old members never reassigned.
    for arrival in data['arrivals']:
        x=arrival['key_features'].numpy()[:,44:].copy();norm=np.linalg.norm(x,axis=1,keepdims=True)
        if not np.isfinite(x).all() or np.any(norm<1e-12):raise ValueError('finite nonzero K features required')
        x/=norm
        updates={}
        for offset,value in enumerate(x):
            token=arrival['frame']*frame_tokens+offset;bucket=sum(int(v>=0)<<j for j,v in enumerate(value[:6]))
            candidates=buckets.get(bucket,[])[-16:];evaluations+=len(candidates)
            available=[g for g in candidates if groups[g]['count']<128]
            scored=[(float(value@(groups[g]['sum']/np.linalg.norm(groups[g]['sum']))),g) for g in available]
            score,g=max(scored,key=lambda pair:(pair[0],-pair[1])) if scored else (-1.,None)
            reason='new'
            if g is not None and score>=.97:
                trial=groups[g]['sum']+value;n=groups[g]['count']+1
                if 1-float(trial@trial)/(n*n)>.03:reason='dispersion_split';g=None
            else:
                reason='capacity_split' if candidates and not available else 'new';g=None
            if g is None:
                g=len(groups);groups.append({'sum':np.zeros_like(value),'count':0});buckets.setdefault(bucket,[]).append(g)
                if reason!='new':splits.append({'frame':arrival['frame'],'group':g,'reason':reason})
            groups[g]['sum']+=value;groups[g]['count']+=1;token_group[token]=g;updates.setdefault(g,[]).append(token)
        assignments.append((arrival['frame'],updates))
    assignment_s=time.perf_counter()-started
    accesses={}
    for event in data['accesses']:accesses.setdefault(event['frame'],[]).append(event)
    results=[]
    for lazy in (False,True):
        runs={};pending={};writes=0;peak=0;flushed=0;forced=0;work_s=0.;checkpoints=[]
        def flush(ids):
            nonlocal writes,flushed,work_s
            began=time.perf_counter()
            for g in ids:
                values=pending.pop(g,[]);target=runs.setdefault(g,[])
                for token in values:
                    if target and target[-1][1]==token:target[-1][1]=token+1;writes+=1
                    else:target.append([token,token+1]);writes+=2
                flushed+=len(values)
            work_s+=time.perf_counter()-began
        for number,(frame,updates) in enumerate(assignments,1):
            for event in accesses.get(frame,[]):
                if any(not 0<=a<b<=frame*frame_tokens for a,b in event['ranges']):raise ValueError('access is not committed history')
                touched={token_group[token] for a,b in event['ranges'] for token in range(a,b)}
                flush(touched&pending.keys())
            for g,values in updates.items():pending.setdefault(g,[]).extend(values)
            backlog=sum(map(len,pending.values()));peak=max(peak,backlog)
            if not lazy or backlog>=2*8*frame_tokens:
                forced+=int(lazy);flush(list(pending))
            if number in (8,16,32):checkpoints.append({'chunks':number,'dirty_tokens':sum(map(len,pending.values())),
                'directory_intervals':sum(map(len,runs.values())),'maintenance_s':work_s,'integer_writes':writes})
        deferred=sum(map(len,pending.values()));before=work_s;flush(list(pending))
        results.append({'variant':'delayed_same_membership' if lazy else 'eager_incremental_same_membership',
            'maintenance_s_including_drain':work_s,'final_drain_s':work_s-before,'dirty_tokens_before_drain':deferred,
            'peak_dirty_tokens':peak,'forced_flushes':forced,'index_integer_writes':writes,
            'final_intervals':sum(map(len,runs.values())),'checkpoints':checkpoints,'raw_KV_bytes_read_for_layout':0})
    started=time.perf_counter();baseline=[]
    for arrival in data['arrivals']:
        for f in range(arrival['frame'],arrival['frame']+8):
            baseline.extend((f*frame_tokens+a,min((f+1)*frame_tokens,f*frame_tokens+a+64)) for a in range(0,frame_tokens,64))
    baseline_s=time.perf_counter()-started
    return {'status':'pass','real_chunks':len(assignments),'raw_tokens':len(token_group),'block64_groups':len(baseline),
        'block64_incremental_directory_s':baseline_s,'K_feature_groups':len(groups),'K_assignment_CPU_s':assignment_s,
        'candidate_centroid_evaluations':evaluations,'capacity_or_dispersion_splits':splits,'maintenance':results,
        'input_feature_bytes':sum(a['key_features'].numel()*a['key_features'].element_size() for a in data['arrivals']),
        'definition':'head-mean committed K, discard44 temporal RoPE channels, unit norm;6sign buckets;last16 candidates;cos>=.97;capacity128;variance<=.03;append-only splits',
        'layout_comparison_same_memberships_and_original_access_ranges':True,
        'limitations':['no raw KV prototype replacement','feature assignment/index maintenance only; full grouped scoring not measured',
            '32 chunks unmeasured if fewer real arrivals exist; no duplicated history','source is a sparse own trajectory, not a dense quality oracle']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    summary=json.loads((args.case/'summary.json').read_text());assert summary['status']=='pass'
    data=torch.load(args.case/'wave2_diagnostics.pt',weights_only=True,map_location='cpu')
    (args.output/'registration.json').write_text(json.dumps({'source':str(args.case),'cosine':.97,'variance':.03,'capacity':128,'candidates_per_bucket':16,'backlog_cap_chunks':2,'no_old_member_reassignment':True},indent=2)+'\n')
    result=replay(data,data['call']['frame_tokens']);(args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('capacity_or_dispersion_splits','maintenance')}))


if __name__=='__main__':main()
