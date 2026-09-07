#!/usr/bin/env python3
"""Complete-capture pre-registered query-conditioned representation-risk screen.

Ten raw25 selectors are built before reading Dense output: legacy, two random
controls and seven compact-moment proxies. All retain the same four-slot
K-feature tail representation, so approximation bytes are directly comparable.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.committed_moments import build_frame_moments,gather_moment_payload,subtract_admitted_raw
from adapters.longlive_sparse.group_relations import summarize_groups
from adapters.longlive_sparse.moment_risk import OnlineMomentContext,score_raw_moment_risk,RISK_CANDIDATES
from adapters.longlive_sparse.offline_eval import dense_history_attention,output_error_metrics
from scripts.evaluate_complete_attention_capture import construct_routes
from scripts.probe_prototype_tail import selected_raw_indices,weighted_attention


def block_table(frame_tokens,frames,block_tokens=64):
    rows=[]
    for f in range(frames):
        for start in range(0,frame_tokens,block_tokens):
            row=torch.full((block_tokens,),-1,dtype=torch.long)
            count=min(block_tokens,frame_tokens-start)
            row[:count]=torch.arange(f*frame_tokens+start,f*frame_tokens+start+count)
            rows.append(row)
    return torch.stack(rows)


def choose_blocks(scores,table,budget):
    choices=[]
    for b in range(scores.shape[0]):
        heads=[]
        for h in range(scores.shape[1]):
            indices=table[torch.argsort(scores[b,h],descending=True,stable=True)].reshape(-1)
            heads.append(indices[indices>=0][:budget].sort().values)
        choices.append(torch.stack(heads))
    return torch.stack(choices)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--capture-manifest',type=Path,required=True)
    p.add_argument('--kind',choices=('motion','state'),required=True);p.add_argument('--layers',default='0,9,19,29')
    p.add_argument('--query-resolution',choices=('groups','tokens'),default='groups',
                   help='GPU token-level prototype scoring tests loss from averaging Q before nonlinear scoring')
    p.add_argument('--query-samples',type=int,default=0,
                   help='0=all current Q; otherwise deterministic uniform token representatives, scored separately before averaging')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    if args.query_samples<0 or (args.query_samples and args.query_resolution!='tokens'):
        raise ValueError('query sampling requires token-level scoring and a nonnegative count')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('real CUDA required')
    manifest=json.loads(args.capture_manifest.read_text())
    paths={r['layer']:r for r in manifest['captures'] if r['kind']==args.kind}
    report=dict(status='running',gpu=torch.cuda.get_device_name(),kind=args.kind,cases=[],
        scope='offline_teacher_evaluation_of_causal_summary_only_scores',formal_promotion=False,
        risk_terms_are_heuristic_not_guaranteed_bounds=True,
        query_resolution=args.query_resolution,
        query_samples=args.query_samples,
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('scripts/probe_moment_risk.py','adapters/longlive_sparse/moment_risk.py','adapters/longlive_sparse/committed_moments.py')})
    try:
        for layer in map(int,args.layers.split(',')):
            source=paths[layer];path=(args.capture_manifest.parent/source['file']).resolve()
            if not path.is_relative_to(args.capture_manifest.parent.resolve()):raise ValueError('capture escapes bundle')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=source['sha256']:raise ValueError('capture hash mismatch')
            capture=torch.load(path,map_location='cpu',weights_only=True)
            raw_route=construct_routes(capture,device='cuda',value_candidates=())['legacy_cap25']
            selected=selected_raw_indices(raw_route,capture['frame_ids'],capture['token_ids'])
            q,k,v,ek,ev=[capture[name].cuda() for name in ('query','key','value','exact_key','exact_value')]
            batch,total,heads,dim=k.shape;frame_tokens=capture['spatial_height']*capture['spatial_width']
            frames=total//frame_tokens;budget=selected.shape[-1]
            ids=[int(capture['frame_ids'][0,0,i*frame_tokens]) for i in range(frames)]
            source_frames=[];km=[];kv=[];vm=[];vv=[];counts=[]
            started=time.perf_counter()
            for f in range(frames):
                fk=k[:,f*frame_tokens:(f+1)*frame_tokens];fv=v[:,f*frame_tokens:(f+1)*frame_tokens]
                moment=build_frame_moments(fk,fv,grouping='key_kmeans',block_tokens=64,groups=4)
                labels=moment.token_groups.long()[...,None].expand(-1,-1,-1,dim)
                kmean=moment.key_sum/moment.counts.clamp_min(1)[...,None]
                vmean=moment.value_sum/moment.counts.clamp_min(1)[...,None]
                k2=torch.zeros_like(moment.key_sum).scatter_add_(2,labels,fk.permute(0,2,1,3).float().square())/moment.counts.clamp_min(1)[...,None]
                v2=torch.zeros_like(moment.value_sum).scatter_add_(2,labels,fv.permute(0,2,1,3).float().square())/moment.counts.clamp_min(1)[...,None]
                km.append(kmean.cpu());vm.append(vmean.cpu());kv.append((k2-kmean.square()).clamp_min(0).cpu())
                vv.append((v2-vmean.square()).clamp_min(0).sum(-1).cpu());counts.append(moment.counts.cpu())
                source_frames.append(moment.cpu())
            torch.cuda.synchronize();index_s=time.perf_counter()-started
            prototypes=(torch.cat(km,2),torch.cat(kv,2),torch.cat(vm,2),torch.cat(vv,2),torch.cat(counts,2))
            prototype_H2D_bytes=0
            if args.query_resolution=='groups':
                summary=summarize_groups(q,grouping='spatial_quadrants',spatial_height=capture['spatial_height'],spatial_width=capture['spatial_width'])
                q2=[];qlabel=summary.query_labels.cuda();qvalues=q.permute(0,2,1,3).float().square()
                for g in range(4):
                    mask=(qlabel==g).float()
                    q2.append((qvalues*mask[...,None]).sum(2)/mask.sum(-1).clamp_min(1)[...,None])
                q2=torch.stack(q2,2).cpu()
                context=OnlineMomentContext(summary.query_centroids,q2,summary.query_group_sizes,*prototypes)
            else:
                # Full current Q remains on GPU; only committed prototypes are uploaded.
                prototype_H2D_bytes=sum(t.numel()*t.element_size() for t in prototypes)
                current_q=q.permute(0,2,1,3).float()
                if args.query_samples and args.query_samples<current_q.shape[2]:
                    representatives=torch.linspace(0,current_q.shape[2]-1,args.query_samples,device=q.device).round().long()
                    current_q=current_q.index_select(2,representatives)
                context=OnlineMomentContext(current_q,current_q.square(),torch.ones(current_q.shape[:-1],device='cuda'),
                    *(t.cuda() for t in prototypes))
            table=block_table(frame_tokens,frames)
            gen=torch.Generator().manual_seed(20260908)
            selections={'legacy25':selected,'random_tokens25':torch.stack([
                torch.randperm(total,generator=gen)[:budget].sort().values for _ in range(batch*heads)]).reshape(batch,heads,budget),
                'random_blocks25':choose_blocks(torch.rand(batch,heads,table.shape[0],generator=gen),table,budget)}
            score_times={}
            for candidate in RISK_CANDIDATES:
                torch.cuda.synchronize();score_begin=time.perf_counter()
                scores=score_raw_moment_risk(context,candidate=candidate)
                selections[candidate]=choose_blocks(scores.cpu(),table,budget)
                score_times[candidate]=time.perf_counter()-score_begin
            # No selector or proxy is evaluated after reading teacher output.
            reference=dense_history_attention(q,torch.cat((ek,k),1),torch.cat((ev,v),1))
            results={};evaluated={}
            for name,indices in selections.items():
                if indices.shape!=selected.shape:raise ValueError('raw budget changed')
                coordinate_sha=hashlib.sha256(indices.contiguous().numpy().tobytes()).hexdigest()
                if coordinate_sha in evaluated:
                    other=evaluated[coordinate_sha]
                    results[name]=dict(results[other],identical_selection_to=other)
                    continue
                frame_ids=capture['frame_ids'].gather(2,indices)
                token_ids=capture['token_ids'].gather(2,indices)
                payload=gather_moment_payload(source_frames,ids,frame_ids,token_ids)
                index=indices.cuda()[...,None].expand(-1,-1,-1,dim)
                hk=k.permute(0,2,1,3).gather(2,index).permute(0,2,1,3)
                hv=v.permute(0,2,1,3).gather(2,index).permute(0,2,1,3)
                tail=subtract_admitted_raw(tuple(t.cuda() for t in payload),hk,hv,frame_tokens=frame_tokens,block_tokens=64)
                output=torch.empty_like(q,dtype=torch.float32)
                for b in range(batch):
                    for h in range(heads):
                        ak=torch.cat((ek[b,:,h],hk[b,:,h],tail.key[b,:,h]))
                        av=torch.cat((ev[b,:,h],hv[b,:,h],tail.value[b,:,h]))
                        ac=torch.cat((torch.ones(ek.shape[1]+budget,device='cuda'),tail.counts[b,h]))
                        output[b,:,h]=weighted_attention(q[b,:,h],ak,av,ac)
                results[name]=dict(error=output_error_metrics(reference,output),raw_tokens_per_head=budget,
                    virtual_prototype_slots=tail.key.shape[1],raw_plus_BF16_prototype_bytes=(hk.numel()+hv.numel())*2+tail.bytes,
                    selected_coordinate_sha256=coordinate_sha)
                evaluated[coordinate_sha]=name
            row=dict(layer=layer,capture_sha256=source['sha256'],index_reconstruction_s=index_s,results=results,
                     all_selections_before_teacher=True,full_exact_current_recent_teacher=True,equal_raw_and_representation_slots=True,
                     query_resolution=args.query_resolution,prototype_H2D_bytes_for_scoring=prototype_H2D_bytes,
                     scored_query_count=context.query_mean.shape[2],
                     score_and_selection_wall_s=score_times,scoring_times_are_unwarmed_diagnostics=True)
            report['cases'].append(row);(args.output/f'layer{layer:02d}.json').write_text(json.dumps(row,indent=2)+'\n')
            print(json.dumps(dict(layer=layer,relative_l2={name:r['error']['relative_l2'] for name,r in results.items()})),flush=True)
        report['status']='pass'
    except BaseException:
        report.update(status='fail',traceback=traceback.format_exc());raise
    finally:(args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
