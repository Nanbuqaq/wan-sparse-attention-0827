#!/usr/bin/env python3
"""Offline fixed-route deletion on one common full-source captured trajectory."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.history_cache import tensor_sha256
from adapters.longlive_sparse.native_causal_block_memory import groups_for_source,source_head_scores,exact_source_indices
from scripts.analyze_native_attention_teacher import output_error


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True)
    p.add_argument('--projection-checkpoint',type=Path)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('real GPU required')
    torch.backends.cuda.matmul.allow_tf32=False
    summary=json.loads((args.case/'summary.json').read_text())
    if (summary['status']!='pass' or not summary['observer_noise_latent_RGB_equivalence']
        or summary['causal_block_config']['policy']!='full' or summary['seed']!=20260913):
        raise ValueError('existing-output-equivalent toy13 full-source teacher required')
    capture_path=args.case/'attention_teacher.pt'
    if sha(capture_path)!=summary['attention_teacher']['sha256']:raise ValueError('teacher artifact SHA mismatch')
    data=torch.load(capture_path,weights_only=True,map_location='cpu',mmap=True)
    if not data.get('online_routing_may_not_access'):raise ValueError('offline teacher boundary missing')
    projection_state=None;projection_sha=None
    if args.projection_checkpoint:
        projection_sha=sha(args.projection_checkpoint)
        asset_path=args.projection_checkpoint.parent.parent/'assets_manifest.json'
        if sha(asset_path)!=summary['assets_manifest_sha256']:
            raise ValueError('projection checkpoint assets differ from the captured model')
        asset=json.loads(asset_path.read_text())
        matches=[r for r in asset['files'] if r['file']==args.projection_checkpoint.name]
        if len(matches)!=1 or matches[0]['sha256']!=projection_sha:
            raise ValueError('projection checkpoint does not match the locked model weight hash')
        projection_state=torch.load(args.projection_checkpoint,weights_only=True,map_location='cpu',mmap=True)['generator']
        for layer in (0,14,29):
            if projection_state[f'model.blocks.{layer}.self_attn.o.weight'].shape!=(3072,3072):
                raise ValueError('native output projection geometry differs')
    spec=json.loads(args.manifest.read_text());mask_path=Path(spec['source_mask'])
    mask=torch.load(mask_path,weights_only=True,map_location='cpu')
    latent=torch.load(args.case/'latents.pt',weights_only=True,map_location='cpu')
    if tensor_sha256(latent[:,mask['source_start']:mask['source_end']])!=mask['source_latent_sha256']:
        raise ValueError('teacher source differs from the past-only geometry mask')
    del latent
    live={};provenance=[]
    for arm in spec['live_routes']:
        directory=Path(arm['case']);d=json.loads((directory/'summary.json').read_text())
        for field in ('seed','noise_sha256','pre_return_latent_sha256','gpu','assets_manifest_sha256'):
            if d[field]!=summary[field]:raise ValueError('fixed-route source identity differs: '+field)
        path=directory/'causal_block_routes.pt'
        if d['status']!='pass' or sha(path)!=d['causal_block_routes']['sha256']:raise ValueError('route status/hash mismatch')
        records=torch.load(path,weights_only=True,map_location='cpu')['records']
        if (len(records)!=30 or {(r['layer'],r.get('phase',0)) for r in records}!={(i,0) for i in range(30)}):
            raise ValueError('fixed first-return routes required')
        for row in records:
            ids=row['source_indices']
            if (row['frame']!=96 or row['source_start']!=40 or row['destination_start']!=7040
                or ids.shape!=(24,1760) or ids.min()<0 or ids.max()>=7040 or not torch.all(ids[:,1:]>ids[:,:-1])):
                raise ValueError('unqualified source route geometry')
        live[arm['id']]={r['layer']:r['source_indices'] for r in records}
        provenance.append(dict(id=arm['id'],case=str(directory),route_sha256=sha(path),
            route_kind='actual first-return live selection transplanted onto full-source teacher Q'))
    protected=torch.tensor(list(range(7040))+list(range(14080,28160)),device='cuda')
    heads=torch.arange(24,device='cuda')[:,None];rows=[];gates=[]
    for ri,record in enumerate(data['records']):
        if (record['query_frame']!=96 or record['q'].shape!=(1,32,24,128)
            or record['k'].shape!=(1,28160,24,128)):
            raise ValueError('unexpected native teacher geometry')
        q,k,v=[record[n].cuda() for n in ('q','k','v')]
        qh=q[0].permute(1,0,2).float();kh=k[0].permute(1,0,2).float();vh=v[0].permute(1,0,2).float()
        probability=(qh@kh.transpose(-1,-2)*(128**-.5)).softmax(-1)
        full=probability@vh
        projected_full=projection_weight=projection_bias=None
        if projection_state is not None:
            prefix=f'model.blocks.{record["layer"]}.self_attn.o.'
            projection_weight=projection_state[prefix+'weight'].cuda().float()
            projection_bias=projection_state[prefix+'bias'].cuda().float()
            projected_full=full.transpose(0,1).reshape(32,3072)@projection_weight.T+projection_bias
        error=output_error(full,record['native_output'][0].permute(1,0,2).cuda())
        gate=error['max_abs']<=.02 and error['relative_l2']<=.01 and error['one_minus_cosine']<=.001
        gates.append(dict(capture_row=ri,layer=record['layer'],phase=record['phase'],pass_gate=gate,error=error))
        arms={name:route[record['layer']].long().cuda() for name,route in live.items()}
        groups=groups_for_source(22,40,kind='flat_tube_matched')
        indices=torch.tensor(groups,device='cuda')+7040
        km,vm=[raw[0].index_select(0,indices.flatten()).float().reshape(110,64,24,128).mean(1) for raw in (k,v)]
        scores=source_head_scores(q[0],km,vm,torch.full((110,),64,device='cuda'),'mass_value',32).cpu().tolist()
        arms['flat64_from_capture_Q']=torch.tensor([exact_source_indices(groups,s,1760) for s in scores],device='cuda')
        arms['full_source']=torch.arange(7040,device='cuda')[None].expand(24,-1)
        for name,selected in arms.items():
            keep=torch.cat([protected[None].expand(24,-1),selected+7040],dim=1).sort(1).values
            kk,vv=kh[heads,keep],vh[heads,keep]
            remaining=(qh@kk.transpose(-1,-2)*(128**-.5)).softmax(-1)@vv
            diff=(remaining-full).square()
            per_query=(diff.sum((0,2))/full.square().sum((0,2)).clamp_min(1e-30)).sqrt()
            kept_mass=probability.gather(2,(selected+7040)[:,None,:].expand(-1,32,-1)).sum(-1)
            source_mass=probability[:,:,7040:14080].sum(-1)
            rows.append(dict(capture_row=ri,layer=record['layer'],phase=record['phase'],method=name,
                source_tokens_per_head=selected.shape[1],native_reference_gate=gate,
                actual_deleted_output_error=output_error(full,remaining),
                per_query_relL2=per_query.cpu().tolist(),query_indices=record['query_indices'].tolist(),
                mean_retained_source_attention_mass_fraction=float(kept_mass.sum()/source_mass.sum()),
                route_kind='offline recomputed from capture Q' if name=='flat64_from_capture_Q' else
                    ('no deletion control' if name=='full_source' else 'fixed executed live graph on common teacher trajectory')))
            if projected_full is not None:
                projected_remaining=remaining.transpose(0,1).reshape(32,3072)@projection_weight.T+projection_bias
                rows[-1]['after_native_output_projection_FP32_error']=output_error(projected_full,projected_remaining)
            print(json.dumps({k:rows[-1][k] for k in ('capture_row','method','actual_deleted_output_error','native_reference_gate')}),flush=True)
            del kk,vv,remaining
        del q,k,v,qh,kh,vh,probability,full,arms
    report=dict(status='offline_complete',rows=rows,native_reference_gates=gates,
        capture_sha256=sha(capture_path),manifest_sha256=sha(args.manifest),source_mask_sha256=sha(mask_path),
        actual_source_latent_matches_mask=True,route_provenance=provenance,GPU=torch.cuda.get_device_name(),
        projection_checkpoint_sha256=projection_sha,
        projection_scope='FP32 application of original BF16 learned output weights and bias; no residual/FFN or native BF16 linear-rounding model' if projection_state is not None else None,
        limits=['same full-source trajectory for fixed graph deletion, not actual per-method closed-loop output error',
            'old32 geometric Q samples only, not full-Q and not object-grounded current-query labels',
            'mask is offline privileged diagnostic only; no online route may access it',
            'one toy13 source, no new independent quality sample or causal proof from correlation',
            'failed native reference rows remain failed; no threshold relaxation or layer selection'])
    (args.output/'fixed_route_teacher.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
