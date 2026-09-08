#!/usr/bin/env python3
"""CPU-only role mass/contribution diagnostics on verified actual Q/K/V/O.

Local output sensitivity is not a semantic accuracy metric or an online score.
The exact group logsumexp decomposition avoids subtracting nearly equal masses.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def summary(x):
    x=x.float().flatten()
    return dict(mean=float(x.mean()),p50=float(x.median()),p95=float(torch.quantile(x,.95)),
                min=float(x.min()),max=float(x.max()))


def output_error(reference,actual):
    r=reference.float().flatten();a=actual.float().flatten();nr=r.norm();na=a.norm()
    cosine=0. if float(nr)==0 and float(na)==0 else float(1-torch.dot(r,a)/(nr*na).clamp_min(1e-12))
    return dict(max_abs=float((r-a).abs().max()),relative_l2=float((r-a).norm()/nr.clamp_min(1e-12)),one_minus_cosine=cosine)


@torch.inference_mode()
def decompose(record,labels):
    q=record['q'][0].permute(1,0,2).float().contiguous()
    native=record['native_output'][0].permute(1,0,2).float().contiguous()
    width=8*record['frame_tokens']
    if record['k'].shape[1]!=width*len(labels) or record['v'].shape!=record['k'].shape:
        raise ValueError('actual K/V do not match declared role partition')
    if record.get('attention_causal_mask',False):raise ValueError('unregistered per-query causal mask')
    role_outputs=[];log_z=[]
    for index,label in enumerate(labels):
        sl=slice(index*width,(index+1)*width)
        k=record['k'][0,sl].permute(1,0,2).float().contiguous()
        v=record['v'][0,sl].permute(1,0,2).float().contiguous()
        logits=torch.bmm(q,k.transpose(1,2))*record['softmax_scale']
        log_z.append(torch.logsumexp(logits,dim=-1))
        role_outputs.append(torch.bmm(torch.softmax(logits,dim=-1),v))
    z=torch.stack(log_z,dim=-1);values=torch.stack(role_outputs,dim=-2)
    mass=torch.softmax(z,dim=-1);full=(mass.unsqueeze(-1)*values).sum(-2)
    error=output_error(full,native)
    gate=error['max_abs']<=.02 and error['relative_l2']<=.01 and error['one_minus_cosine']<=.001
    roles={};norm=full.norm(dim=-1).clamp_min(1e-12)
    for i,label in enumerate(labels):
        others=[j for j in range(len(labels)) if j!=i]
        dropped=(torch.softmax(z[...,others],-1).unsqueeze(-1)*values[...,others,:]).sum(-2)
        contribution=mass[...,i,None]*values[...,i,:]
        roles[label]=dict(probability_mass=summary(mass[...,i]),
            contribution_norm_over_total=summary(contribution.norm(dim=-1)/norm),
            remove_role_output_sensitivity=summary((dropped-full).norm(dim=-1)/norm),
            mass_by_query_site={site:summary(mass[:,j::4,i]) for j,site in enumerate(record.get('query_sites',[]))})
    source=labels.index('source');gains={}
    for gain in (1.,2.,4.,8.):
        biased=z.clone();biased[...,source]+=torch.tensor(gain).log()
        weighted_mass=torch.softmax(biased,-1)
        changed=(weighted_mass.unsqueeze(-1)*values).sum(-2)
        gains[str(gain)]=dict(source_mass=summary(weighted_mass[...,source]),
            relative_output_change=summary((changed-full).norm(dim=-1)/norm))
    return dict(query_frame=record['query_frame'],phase=record['phase'],layer=record['layer'],
        original_Q_tokens=record['full_Q_tokens'],sampled_queries_per_head=q.shape[1],heads=q.shape[0],
        K_tokens=record['k'].shape[1],FP32_replay_error=error,FP32_replay_gate=gate,
        roles=roles,offline_source_gain_sensitivity_not_quality=gains),dict(labels=labels,log_z=z,mass=mass,role_outputs=values,fp32_output=full,native_output=native)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-unverified-capture',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.set_num_interop_threads(1)
    reports=[]
    for arm,labels in (('shot',['initial','source','away','current']),('reset_reveal',['initial','source','current'])):
        root=args.root/arm;source=json.loads((root/'summary.json').read_text())
        verified=source['status']=='pass' and source.get('observer_noise_latent_RGB_equivalence',False)
        if not verified and not args.allow_unverified_capture:raise ValueError('full trajectory capture equivalence has not passed')
        path=root/'attention_teacher.pt'
        with path.open('rb') as handle:sha=hashlib.file_digest(handle,'sha256').hexdigest()
        if verified and sha!=source['attention_teacher']['sha256']:raise ValueError('capture payload SHA differs')
        data=torch.load(path,map_location='cpu',weights_only=True)
        if data['schema']!='native_attention_teacher_v1' or not data['online_routing_may_not_access']:raise ValueError('wrong capture context')
        rows=[];tensors=[]
        for record in data['records']:
            row,derived=decompose(record,labels);rows.append(row);tensors.append(derived)
        torch.save(tensors,args.output/f'{arm}__group_decomposition.pt')
        report=dict(arm=arm,full_trajectory_equivalence_verified=verified,capture_sha256=sha,rows=rows,
                    all_FP32_replay_gates=all(r['FP32_replay_gate'] for r in rows))
        reports.append(report)
        print(json.dumps(dict(arm=arm,records=len(rows),verified=verified,FP32_gate=report['all_FP32_replay_gates'])),flush=True)
        del data,tensors
    result=dict(schema='native_attention_role_diagnostics_v1',device='CPU',cases=reports,
        diagnostic_only_not_semantic_quality=True,online_proxy_not_frozen=True,query_sites_are_geometric_not_semantic_masks=True)
    (args.output/'role_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
