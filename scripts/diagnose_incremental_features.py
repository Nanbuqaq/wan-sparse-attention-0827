#!/usr/bin/env python3
"""Causal nearest-old-prototype diagnosis, separating feature and candidate gates."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch


def diagnose(features):
    x=np.asarray(features,dtype=np.float32);x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    if not np.isfinite(x).all():raise ValueError('invalid feature')
    sites=set(np.linspace(1,len(x)-1,min(256,len(x)-1)).round().astype(int));buckets={};rows=[]
    for i,value in enumerate(x):
        key=sum(int(v>=0)<<bit for bit,v in enumerate(value[:6]));candidates=buckets.get(key,[])[-16:]
        if i in sites:
            scores=x[:i]@value;best=int(np.argmax(scores));exact=float(scores[best])
            limited=max((float(x[c]@value) for c in candidates),default=-1.)
            rows.append(dict(position=i,exact_previous=best,exact_cos=exact,bucket_last16_cos=limited,
                exact_in_candidates=best in candidates,candidates=len(candidates)))
        buckets.setdefault(key,[]).append(i)
    exact=np.array([r['exact_cos'] for r in rows]);limited=np.array([r['bucket_last16_cos'] for r in rows])
    return dict(samples=len(rows),raw_rows=rows,exact_cos={str(p):float(np.quantile(exact,p)) for p in (.5,.9,.99)},
        exact_above_097_fraction=float(np.mean(exact>=.97)),restricted_above_097_fraction=float(np.mean(limited>=.97)),
        nearest_candidate_recall=float(np.mean([r['exact_in_candidates'] for r in rows])),mean_candidate_score_loss=float(np.mean(exact-limited)),
        rejected_despite_exact_above_threshold=int(np.sum((exact>=.97)&(limited<.97))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--observer',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.perf_counter()
    data=torch.load(a.case/('steady_observer.pt' if a.observer else 'wave2_diagnostics.pt'),weights_only=True,map_location='cpu',mmap=True)
    arrivals=data['arrivals'];definitions={}
    if a.observer:
        full=torch.cat([x['key_mean'] for x in arrivals]);mean=full.mean(1)
        for name,values in [('per_head_concat_post_rope',full),('per_head_concat_drop44',full[:,:,44:])]:
            definitions[name]=(torch.nn.functional.normalize(values.float(),dim=-1).flatten(1)/math.sqrt(values.shape[1])).numpy()
        atoms=arrivals[0]['group_atom_tokens']
    else:mean=torch.cat([x['key_features'] for x in arrivals]);atoms=1
    definitions['head_mean_drop44']=mean[:,44:].numpy()
    rows={name:diagnose(x) for name,x in definitions.items()}
    result=dict(status='pass',source=str(a.case),real_chunks=len(arrivals),atom_original_tokens=atoms,rows=rows,wall_s=time.perf_counter()-start,
        scope='sampled causal exact NN among all previous singleton atom prototypes; isolates initial assignment gate, not final clustered-centroid quality',
        limitations=['drop44 retains spatial RoPE, not position invariant','old head-mean capture cannot recover independent heads','no video quality used for thresholds'])
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({name:{k:v for k,v in row.items() if k!='raw_rows'} for name,row in rows.items()}),flush=True)


if __name__=='__main__':main()
