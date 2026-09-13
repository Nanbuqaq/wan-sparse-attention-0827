#!/usr/bin/env python3
"""One source-calibrated, stable-ID incremental reference on real Block16 arrivals."""
import argparse,json,math,time,hashlib
from pathlib import Path
import numpy as np
import torch
from diagnose_incremental_features import diagnose


def access_atoms(event,frame_tokens):
    frame=event['frame'];apf=frame_tokens//16
    if 'full_visible_owners' in event:
        frames=[x[1] for x in event['full_visible_owners'] if x[0]=='native' and x[1]<frame]
        return {a for f in frames for a in range(f*apf,(f+1)*apf)}
    chosen=event['selected_group_mask'].any(0).nonzero().flatten().tolist()
    groups_per_frame=(frame_tokens+63)//64;result=set(range(min(frame,8)*apf))
    for group in chosen:
        f,b=divmod(group,groups_per_frame);owner=event['eligible'][f][1]
        if owner[0]!='native':raise ValueError('continuous reference cannot reinterpret recalled bindings')
        start=owner[1]*frame_tokens+b*64;stop=owner[1]*frame_tokens+min((b+1)*64,frame_tokens)
        result.update(range(start//16,stop//16))
    if result and max(result)>=frame*apf:raise ValueError('future access')
    return result


def maintenance(assignments,events,frame_tokens,lazy):
    pending={};runs={};writes=0;peak=0;forced=0;elapsed=0.;apc=8*frame_tokens//16
    def flush(ids):
        nonlocal writes,elapsed
        start=time.perf_counter()
        for g in ids:
            target=runs.setdefault(int(g),[])
            for atom in pending.pop(int(g),[]):
                begin=atom*16
                if target and target[-1][1]==begin:target[-1][1]+=16;writes+=1
                else:target.append([begin,begin+16]);writes+=2
        elapsed+=time.perf_counter()-start
    checkpoints=[]
    for chunk in range(len(assignments)//apc):
        frame=chunk*8
        for atoms in events.get(frame,[]):
            touched={int(assignments[a]) for a in atoms};flush(touched&pending.keys())
        for atom in range(chunk*apc,(chunk+1)*apc):pending.setdefault(int(assignments[atom]),[]).append(atom)
        backlog=sum(map(len,pending.values()));peak=max(peak,backlog)
        if not lazy or backlog>=2*apc:forced+=int(lazy);flush(list(pending))
        if chunk+1 in (8,16):checkpoints.append(dict(chunks=chunk+1,dirty_atoms=sum(map(len,pending.values())),maintenance_s=elapsed))
    debt=sum(map(len,pending.values()));start=elapsed;flush(list(pending))
    digest=hashlib.sha256(json.dumps(runs,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(lazy=lazy,maintenance_s_including_final_drain=elapsed,final_drain_s=elapsed-start,dirty_atoms_before_final_drain=debt,
        peak_dirty_atoms=peak,forced_flushes=forced,integer_writes=writes,intervals=sum(map(len,runs.values())),directory_sha256=digest,checkpoints=checkpoints)


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    registration=dict(feature='per-head unit-normalized concat after dropping44 temporal channels; spatial RoPE retained',atoms=16,
        calibration='first4 real clean chunks only; first4 chunks retain causal Block64 membership',join='median exact previous-atom NN cosine in source calibration',
        dispersion='source Block64 dispersion p95',capacity_raw_tokens=128,stable_IDs=True,old_members_never_reassigned=True,
        assignment_search='exact all currently open centroids, reference cost fully measured',lazy_backlog_cap_chunks=2,
        no_video_quality_threshold_tuning=True,source=str(a.case))
    (a.output/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    summary=json.loads((a.case/'summary.json').read_text())
    if len(summary['segments'])!=1:raise ValueError('first registered reference is continuous only')
    data=torch.load(a.case/'steady_observer.pt',weights_only=True,map_location='cpu',mmap=True);arrivals=data['arrivals'];ft=arrivals[0]['frame_tokens'];apf=ft//16;apc=8*apf
    if [x['frame'] for x in arrivals]!=list(range(0,len(arrivals)*8,8)) or len(arrivals)!=16:raise ValueError('expected real16 sequential chunks')
    keys=torch.cat([x['key_mean'] for x in arrivals])[:,:,44:]
    x=(torch.nn.functional.normalize(keys.float(),dim=-1).flatten(1)/math.sqrt(keys.shape[1])).numpy();n,dim=x.shape;calib=4*apc
    started=time.perf_counter();distance=diagnose(x[:calib]);threshold=distance['exact_cos']['0.5']
    baseline=np.array([(i//apf)*((ft+63)//64)+(i%apf)//4 for i in range(n)],dtype=np.int64)
    source_groups=int(baseline[calib-1])+1;disp=[]
    for group in range(source_groups):
        z=x[:calib][baseline[:calib]==group].mean(0);disp.append(1-float(z@z))
    limit=float(np.quantile(disp,.95));calibration_s=time.perf_counter()-started
    sums=np.zeros((n,dim),np.float32);centroids=np.zeros_like(sums);counts=np.zeros(n,np.int32);ids=np.empty(n,np.int64);ng=0;joins=0;rejected={'threshold':0,'dispersion':0};checkpoints=[]
    started=time.perf_counter()
    for i,value in enumerate(x):
        if i<calib:g=int(baseline[i]);ng=max(ng,g+1)
        else:
            scores=centroids[:ng]@value;scores[counts[:ng]>=8]=-np.inf;g=int(np.argmax(scores));best=float(scores[g])
            if best<threshold:rejected['threshold']+=1;g=ng;ng+=1
            else:
                trial=sums[g]+value;size=int(counts[g])+1
                if 1-float(trial@trial)/(size*size)>limit:rejected['dispersion']+=1;g=ng;ng+=1
                else:joins+=1
        ids[i]=g;sums[g]+=value;counts[g]+=1;centroids[g]=sums[g]/max(float(np.linalg.norm(sums[g])),1e-12)
        if (i+1) in (8*apc,16*apc):checkpoints.append(dict(chunks=(i+1)//apc,groups=ng,Block64_groups=int(baseline[i])+1))
    assignment_s=time.perf_counter()-started
    if counts[:ng].max()>8 or counts[:ng].sum()!=n:raise RuntimeError('capacity or coverage failure')
    events={}
    for event in data['accesses']:events.setdefault(event['frame'],[]).append(access_atoms(event,ft))
    eager=maintenance(ids,events,ft,False);lazy=maintenance(ids,events,ft,True)
    if eager['directory_sha256']!=lazy['directory_sha256']:raise RuntimeError('deferred directory differs after final drain')
    q=data['calls'][-1];eligible_frames={owner[1] for _,owner in q['eligible']};eligible_atoms=[i for f in sorted(eligible_frames) for i in range(f*apf,(f+1)*apf)]
    active_groups=len(set(ids[eligible_atoms]));active_baseline=len(set(baseline[eligible_atoms]))
    result=dict(status='pass',real_chunks=16,raw_tokens=n*16,groups=ng,Block64_groups=int(baseline.max())+1,
        new_atom_joins=joins,new_atom_count=n-calib,post_calibration_singleton_groups=int(np.sum(counts[source_groups:ng]==1)),
        calibrated_join_cos=threshold,calibrated_dispersion_limit=limit,source_only_calibration_s=calibration_s,assignment_CPU_s=assignment_s,
        rejection_counts=rejected,checkpoints=checkpoints,maintenance=[eager,lazy],query88_active_groups=active_groups,query88_Block64_groups=active_baseline,
        feature_CPU_bytes=x.nbytes,centroid_capacity_CPU_bytes=sums.nbytes+centroids.nbytes+counts.nbytes,
        required_new_K_summary_D2H_bytes=sum(t['key_mean'].numel()*t['key_mean'].element_size() for t in arrivals),
        limitations=['reference includes no raw-KV replacement','active group count after resident-domain clipping is separately reported',
            'Block16 feature production/D2H is extra versus existing GPU Block64 summaries; no net system speed claim',
            'exact centroid search is a bounded16-chunk reference, not a proven long-session backend','query/pack/Attention evaluation remains separate'])
    torch.save(dict(atom_group_ids=torch.from_numpy(ids.copy()),Block64_group_ids=torch.from_numpy(baseline),atom_tokens=16,registration=registration),a.output/'memberships.pt')
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
