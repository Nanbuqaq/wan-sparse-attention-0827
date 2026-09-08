#!/usr/bin/env python3
"""Whole-block raw precision plus compact prototypes at explicit byte budgets.

This is isolated capture evidence. Wire/metadata counts are not a measured
onload or video speedup. All choices precede the full-context teacher.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.prototype_wire import encode_prototype_frame,choose_whole_blocks,whole_block_token_indices
from adapters.longlive_sparse.moment_risk import OnlineMomentContext,score_raw_moment_risk
from adapters.longlive_sparse.offline_eval import dense_history_attention,output_error_metrics
from scripts.evaluate_complete_attention_capture import construct_routes
from scripts.probe_prototype_tail import selected_raw_indices,weighted_attention


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--capture-manifest',type=Path,required=True)
    p.add_argument('--kind',choices=('motion','state'),required=True);p.add_argument('--layers',default='0,9,19,29')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('real CUDA required')
    paths={r['layer']:r for r in json.loads(args.capture_manifest.read_text())['captures'] if r['kind']==args.kind}
    report=dict(status='running',kind=args.kind,gpu=torch.cuda.get_device_name(),cases=[],
        scope='complete_capture_precision_budget_probe_not_runtime_or_video',online_teacher_access=False,
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('scripts/probe_precision_budget.py','adapters/longlive_sparse/prototype_wire.py','adapters/longlive_sparse/moment_risk.py')})
    try:
        for layer in map(int,args.layers.split(',')):
            record=paths[layer];path=(args.capture_manifest.parent/record['file']).resolve()
            if not path.is_relative_to(args.capture_manifest.parent.resolve()):raise ValueError('capture escapes bundle')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('capture SHA mismatch')
            capture=torch.load(path,map_location='cpu',weights_only=True)
            legacy=construct_routes(capture,device='cuda',value_candidates=())['legacy_cap25']
            legacy_indices=selected_raw_indices(legacy,capture['frame_ids'],capture['token_ids'])
            q,k,v,ek,ev=[capture[n].cuda() for n in ('query','key','value','exact_key','exact_value')]
            if k.dtype!=torch.bfloat16 or v.dtype!=torch.bfloat16:raise ValueError('this wire-byte screen requires BF16 source KV')
            B,T,H,D=k.shape;F=capture['spatial_height']*capture['spatial_width'];frames=T//F
            per_frame=(F+63)//64
            widths=torch.tensor([min(64,F-start) for _ in range(frames) for start in range(0,F,64)])
            representatives=torch.linspace(0,q.shape[1]-1,min(1024,q.shape[1]),device='cuda').round().long()
            current_q=q.permute(0,2,1,3).index_select(2,representatives).float()
            variants={};wire_rows={}
            for codec in ('bf16','u8_scaled'):
                wires=[encode_prototype_frame(k[:,i*F:(i+1)*F],v[:,i*F:(i+1)*F],variance_codec=codec).cpu() for i in range(frames)]
                # Explicit encoded upload; decode quantized variance only on GPU.
                km=torch.cat([w.key_mean for w in wires],2).cuda()
                vm=torch.cat([w.value_mean for w in wires],2).cuda()
                encoded_var=torch.cat([w.key_variance for w in wires],2).cuda()
                counts=torch.cat([w.counts for w in wires],2).cuda()
                variance=encoded_var.float()
                if codec=='u8_scaled':variance*=torch.cat([w.variance_scale for w in wires],2).cuda()
                context=OnlineMomentContext(current_q,current_q.square(),torch.ones(current_q.shape[:-1],device='cuda'),
                    km,variance,vm,torch.zeros_like(counts,dtype=torch.float32),counts)
                scores=score_raw_moment_risk(context,candidate='mass_key_variance').cpu()
                wire_rows[codec]=dict(bytes=sum(w.bytes for w in wires),key_mean=km,value_mean=vm,counts=counts)
                for density in (.10,.14,.20,.25):
                    blocks,raw_counts=choose_whole_blocks(scores,widths,token_budget=int(T*density),byte_normalized=True)
                    selections=[[whole_block_token_indices(blocks[b][h],frame_tokens=F,frames=frames) for h in range(H)] for b in range(B)]
                    variants[f'{codec}_risk_raw{density:.2f}']=dict(codec=codec,density=density,blocks=blocks,selections=selections,raw_counts=raw_counts)
                if codec=='u8_scaled':
                    gen=torch.Generator().manual_seed(20260908)
                    random_scores=torch.rand(scores.shape,generator=gen)
                    blocks,raw_counts=choose_whole_blocks(random_scores,widths,token_budget=int(T*.14),byte_normalized=False)
                    selections=[[whole_block_token_indices(blocks[b][h],frame_tokens=F,frames=frames) for h in range(H)] for b in range(B)]
                    variants['u8_random_raw0.14']=dict(codec=codec,density=.14,blocks=blocks,selections=selections,raw_counts=raw_counts)
            # Every precision choice exists before the full-context teacher is read.
            reference=dense_history_attention(q,torch.cat((ek,k),1),torch.cat((ev,v),1))
            rows={}
            for name,variant in [('legacy25_drop',None),*variants.items()]:
                output=torch.empty_like(q,dtype=torch.float32)
                prototype_nonzero=0
                for b in range(B):
                    for h in range(H):
                        indices=(legacy_indices[b,h] if variant is None else variant['selections'][b][h]).cuda()
                        keys=[ek[b,:,h],k[b,indices,h]];values=[ev[b,:,h],v[b,indices,h]]
                        weights=[torch.ones(ek.shape[1]+indices.numel(),device='cuda')]
                        if variant is not None:
                            wire=wire_rows[variant['codec']]
                            count=wire['counts'][b,h].float().clone().reshape(frames*per_frame,4)
                            selected_blocks=variant['blocks'][b][h]
                            if selected_blocks:count[torch.tensor(selected_blocks,device='cuda')]=0
                            count=count.flatten()
                            assert int(count.sum())+indices.numel()==T
                            keys.append(wire['key_mean'][b,h]);values.append(wire['value_mean'][b,h]);weights.append(count)
                            prototype_nonzero+=int((count>0).sum())
                        output[b,:,h]=weighted_attention(q[b,:,h],torch.cat(keys),torch.cat(values),torch.cat(weights))
                if variant is None:
                    raw_count=legacy_indices.shape[-1];raw_payload=B*H*raw_count*D*4;wire_bytes=0
                    actual_counts=[[raw_count]*H for _ in range(B)]
                else:
                    actual_counts=variant['raw_counts'].tolist()
                    raw_count=int(variant['raw_counts'].max())
                    raw_payload=B*H*raw_count*D*4
                    wire_bytes=wire_rows[variant['codec']]['bytes']
                rows[name]=dict(error=output_error_metrics(reference,output),per_head_raw_tokens=actual_counts,
                    raw_rectangular_payload_bytes=raw_payload,prototype_and_routing_wire_bytes=wire_bytes,
                    raw_plus_wire_over_full_history=(raw_payload+wire_bytes)/(k.numel()*4),
                    active_virtual_keys=prototype_nonzero,byte_estimate_excludes_RoPE_restore_indices_and_control_metadata=True)
            row=dict(layer=layer,capture_sha256=record['sha256'],rows=rows,whole_blocks_only=True,
                     all_choices_before_teacher=True,full_exact_current_recent_teacher=True,query_samples=1024)
            report['cases'].append(row);(args.output/f'layer{layer:02d}.json').write_text(json.dumps(row,indent=2)+'\n')
            print(json.dumps(dict(layer=layer,metrics={name:(r['error']['relative_l2'],r['raw_plus_wire_over_full_history']) for name,r in rows.items()})),flush=True)
        report['status']='pass'
    except BaseException:
        report.update(status='fail',traceback=traceback.format_exc());raise
    finally:(args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
