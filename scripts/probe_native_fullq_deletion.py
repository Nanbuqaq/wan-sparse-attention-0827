#!/usr/bin/env python3
"""Complete-Q fixed deletion replay using the exact native FA2 numerical interface."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.history_cache import tensor_sha256
from scripts.analyze_native_attention_teacher import output_error


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--manifest',type=Path);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('real GPU required')
    if subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()!='6b36d20ec6f7958d29d11a704dfa64611a9f2572':
        raise ValueError('native source lock differs')
    sys.path.insert(0,str(args.source))
    native=importlib.import_module('wan_5b.modules.attention')
    if not native.FLASH_ATTN_2_AVAILABLE or any(getattr(native,k) for k in ('_USE_FA3','_USE_FA4','_USE_TE_ATTN')):
        raise ValueError('exact original FA2 path required; fallback forbidden')
    summary=json.loads((args.case/'summary.json').read_text())
    if (summary['status']!='pass' or not summary['observer_noise_latent_RGB_equivalence']
        or summary['causal_block_config']['policy']!='full' or summary['attention_teacher']['query_mode']!='full'):
        raise ValueError('full-source full-Q capture with complete-output equivalence required')
    path=args.case/'attention_teacher.pt'
    if sha(path)!=summary['attention_teacher']['sha256']:raise ValueError('capture SHA mismatch')
    data=torch.load(path,weights_only=True,map_location='cpu',mmap=True)
    if not data['online_routing_may_not_access'] or data['query_mode']!='full':raise ValueError('full-Q offline boundary missing')
    target=summary['segments'][-1]['start_latent']
    if len(data['records'])!=9 or {(r['query_frame'],r['phase'],r['layer']) for r in data['records']}!={(target,p,l) for p in (0,3,4) for l in (0,14,29)}:
        raise ValueError('full native capture grid differs')
    routes={};provenance=[];multiplicity={}
    if args.manifest:
        if summary['seed']!=20260913 or summary['latent_shape']!=[1,128,48,44,80]:raise ValueError('toy13 diagnostic required')
        spec=json.loads(args.manifest.read_text());mask=torch.load(spec['source_mask'],weights_only=True,map_location='cpu')
        latent=torch.load(args.case/'latents.pt',weights_only=True,map_location='cpu',mmap=True)
        if tensor_sha256(latent[:,40:48])!=mask['source_latent_sha256']:raise ValueError('source mask and teacher source differ')
        del latent
        for arm in spec['live_routes']:
            directory=Path(arm['case']);d=json.loads((directory/'summary.json').read_text());p=directory/'causal_block_routes.pt'
            if d['status']!='pass' or sha(p)!=d['causal_block_routes']['sha256']:raise ValueError('invalid live route artifact')
            for field in ('seed','noise_sha256','pre_return_latent_sha256','gpu','assets_manifest_sha256'):
                if d[field]!=summary[field]:raise ValueError('route source identity differs: '+field)
            records=torch.load(p,weights_only=True,map_location='cpu')['records']
            if len(records)!=30 or {(r['layer'],r.get('phase',0)) for r in records}!={(i,0) for i in range(30)}:
                raise ValueError('fixed first-return routes required')
            for r in records:
                ids=r['source_indices']
                if (r['frame']!=96 or r['source_start']!=40 or r['destination_start']!=7040
                    or ids.shape!=(24,1760) or ids.min()<0 or ids.max()>=7040 or not torch.all(ids[:,1:]>ids[:,:-1])):
                    raise ValueError('actual source route geometry differs')
            routes[arm['id']]={r['layer']:r['source_indices'] for r in records}
            factor=arm.get('repeat_source_frames',1)
            if factor not in (1,4):raise ValueError('only the registered frame multiplicity is allowed')
            if factor==4:
                for r in records:
                    ids=r['source_indices'];frames=ids.reshape(24,2,880)
                    if (not torch.equal(ids,ids[:1].expand_as(ids))
                        or not torch.all(frames%880==torch.arange(880))
                        or not torch.all(frames//880==(frames//880)[:,:,:1])):
                        raise ValueError('multiplicity applies only to two complete shared source frames')
            multiplicity[arm['id']]=factor
            provenance.append(dict(id=arm['id'],route_sha256=sha(p),case=str(directory),source_frame_multiplicity=factor))
    gates=[];rows=[];query_errors=[]
    for ri,record in enumerate(data['records']):
        q,k,v,expected=[record[n].cuda() for n in ('q','k','v','native_output')]
        if (record['query_mode']!='full' or q.shape!=expected.shape
            or q.shape[1]!=record['full_Q_tokens'] or not torch.equal(record['query_indices'],torch.arange(q.shape[1]))):
            raise ValueError('incomplete or reordered Q capture')
        replay=native.attention(q,k,v,fa_version=2)
        exact=torch.equal(replay,expected)
        gates.append(dict(capture_row=ri,layer=record['layer'],phase=record['phase'],Q_tokens=q.shape[1],
            bitwise_exact=exact,error=output_error(expected,replay)))
        if not exact:
            print(json.dumps(gates[-1]),flush=True)
            del q,k,v,expected,replay
            continue
        if routes:
            if q.shape!=(1,7040,24,128) or k.shape!=(1,28160,24,128):raise ValueError('native original-resolution shape required')
            protected=torch.tensor(list(range(7040))+list(range(14080,28160)),device='cuda')
            heads=torch.arange(24,device='cuda')[:,None]
            site=torch.arange(7040,device='cuda')%880
            regions=((site//40)*4//22)*4+((site%40)*4//40)
            arms=[('full_source',None)]+[(name,by_layer[record['layer']]) for name,by_layer in routes.items()]
            for name,selection in arms:
                if selection is None:actual=replay
                else:
                    selected=selection.long().cuda()+7040
                    factor=multiplicity[name]
                    if factor==4:
                        selected=selected.reshape(24,2,880).repeat_interleave(4,dim=1).reshape(24,7040)
                        keep=torch.cat([protected[:7040][None].expand(24,-1),selected,
                            protected[7040:][None].expand(24,-1)],1)
                    else:keep=torch.cat([protected[None].expand(24,-1),selected],1).sort(1).values
                    kk=k[0].permute(1,0,2)[heads,keep].permute(1,0,2)[None].contiguous()
                    vv=v[0].permute(1,0,2)[heads,keep].permute(1,0,2)[None].contiguous()
                    actual=native.attention(q,kk,vv,fa_version=2)
                row=dict(capture_row=ri,layer=record['layer'],phase=record['phase'],method=name,
                    full_Q_tokens=7040,actual_native_BF16_deletion_error=output_error(expected,actual),regions=[])
                row.update(source_frame_multiplicity=multiplicity.get(name,1),
                    unique_source_tokens_per_head=7040 if selection is None else 1760,
                    visible_source_tokens_per_head=7040 if selection is None else 1760*multiplicity[name],
                    offline_reconstruction_not_executed_live=multiplicity.get(name,1)>1)
                for region in range(16):
                    ids=torch.where(regions==region)[0]
                    row['regions'].append(dict(region=region,Q_tokens=ids.numel(),
                        error=output_error(expected.index_select(1,ids),actual.index_select(1,ids))))
                rows.append(row)
                query_errors.append(dict(capture_row=ri,method=name,
                    squared_error=(expected.float()-actual.float()).square().sum((0,2,3)).cpu(),
                    reference_squared_norm=expected.float().square().sum((0,2,3)).cpu()))
                print(json.dumps({k:row[k] for k in ('capture_row','method','actual_native_BF16_deletion_error')}),flush=True)
            del actual,kk,vv,arms
        del q,k,v,expected,replay
    report=dict(status='pass' if len(gates)==9 and all(g['bitwise_exact'] for g in gates) else 'native_replay_gate_fail',
        native_replay_gates=gates,rows=rows,route_provenance=provenance,capture_sha256=sha(path),
        script_sha256=sha(__file__),manifest_sha256=sha(args.manifest) if args.manifest else None,
        source_mask_sha256=sha(spec['source_mask']) if args.manifest else None,
        captured_GPU=summary['gpu'],torch=torch.__version__,
        GPU=torch.cuda.get_device_name(),backend='original native FA2 varlen, explicit fa_version2, no fallback',
        limits=['new exact-native gate does not relabel old FP32 reference failures',
            'multiplicity arms are offline representative reconstructions; they change attention visibility at fixed unique raw-source budget',
            'fixed executed deletion graphs on common full-source trajectory, not actual per-method closed-loop layer outputs',
            'regions are a predeclared uniform4x4 spatial grid over all8 query frames, not oracle-selected queries',
            'same existing toy13 source, not new independent quality samples or a production sparse-kernel benchmark'])
    if query_errors:torch.save(dict(records=query_errors,scope='per-query FP32 squared reductions'),args.output/'query_errors.pt')
    (args.output/'native_fullq_deletion.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],native_exact=sum(g['bitwise_exact'] for g in gates),deletion_rows=len(rows))),flush=True)
    if report['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
