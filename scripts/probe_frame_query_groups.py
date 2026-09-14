#!/usr/bin/env python3
"""Real 5B full-query replay; no capture, teacher or future outputs route online."""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.longlive_sparse.frame_query_groups import FrameQueryRouter, POLICIES
from adapters.longlive_sparse.query_balanced_value import normalized_values, select_static_once, execute_per_head


def errors(reference, actual):
    delta = actual.float()-reference.float()
    return dict(relative_L2=float((delta.square().sum()/reference.float().square().sum().clamp_min(1e-30)).sqrt()),
                max_abs=float(delta.abs().max()))


def selected_teacher(q, k, v, plan):
    sites = plan['meta']['sites']
    sample_groups = plan['meta']['sample_groups']
    result = torch.empty((len(sites), q.shape[2], q.shape[3]), device=q.device)
    for g in range(plan['groups']):
        sample_ids = torch.arange(len(sites),device=q.device) if plan['groups']==1 else sample_groups[g]
        ids = (plan['frame_ids'][g, :, :, None]*880+plan['meta']['offsets']).flatten(-2)
        heads = torch.arange(q.shape[2],device=q.device)[:,None]
        kk = k[0].permute(1,0,2)[heads,ids].float()
        vv = v[0].permute(1,0,2)[heads,ids].float()
        qq = q[0,sites[sample_ids]].permute(1,0,2).float()
        out = ((qq@kk.transpose(-1,-2))/q.shape[-1]**.5).softmax(-1)@vv
        result[sample_ids] = out.permute(1,0,2)
    return result


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cases', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--cache-prototypes',action='store_true')
    p.add_argument('--reference-routes',type=Path)
    args=p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available(): raise RuntimeError('real GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    records=[]
    for case in args.cases:
        path=case/'steady_observer.pt'
        d=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        with path.open('rb') as f: source_sha=hashlib.file_digest(f,'sha256').hexdigest()
        for capture in d['calls']:
            if capture['frame'] not in (24,88): continue
            if capture['layer']!=14 or capture['frame_tokens']!=880:raise RuntimeError('capture geometry differs')
            q,k,v=[capture[x].cuda() for x in ('q','k','v')]
            km,vm,cnt=[capture[x].cuda() for x in ('key_mean','value_mean','counts')]
            eligible=[i for i,_ in capture['eligible']]
            if any(owner[0]!='native' or owner[1]>=capture['frame'] for _,owner in capture['eligible']):
                raise RuntimeError('optional source not past committed native history')
            protected=sorted({i//880 for i in capture['protected']})
            nblocks=14
            assert len(km)==len(eligible)*nblocks
            summaries=[(km[i*nblocks:(i+1)*nblocks],vm[i*nblocks:(i+1)*nblocks],cnt[i*nblocks:(i+1)*nblocks]) for i in range(len(eligible))]
            outputs={}
            dense=None
            for policy in POLICIES:
                router=FrameQueryRouter(policy,capture['token_grid'])
                times=[]
                base_allocated=torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                for repeat in range(args.repeats+1):
                    torch.cuda.synchronize()
                    began=time.perf_counter()
                    before,after=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    before.record()
                    cache_key=(14,tuple(tuple(owner) for _,owner in capture['eligible'])) if args.cache_prototypes else None
                    plan=router.prepare(q,k,eligible,protected,summaries,880,prototype_key=cache_key)
                    after.record()
                    events=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
                    output=router.execute(q,k,v,plan,timing=events)
                    torch.cuda.synchronize()
                    sample=dict(complete_wall_s=time.perf_counter()-began,
                            prepare_host_s=plan['prepare_host_s'],prepare_stream_ms=before.elapsed_time(after),
                            pack_stream_ms=events[0].elapsed_time(events[1]),
                            FA2_stream_ms=events[1].elapsed_time(events[2]),
                            output_restore_stream_ms=events[2].elapsed_time(events[3]))
                    if repeat:times.append(sample)
                    else:cold_sample=sample
                peak=torch.cuda.max_memory_allocated()
                sites=plan['meta']['sites']
                if dense is None:
                    qq=q[0,sites].permute(1,0,2).float()
                    dense=(((qq@k[0].permute(1,2,0).float())/q.shape[-1]**.5).softmax(-1)@v[0].permute(1,0,2).float()).permute(1,0,2)
                teacher=selected_teacher(q,k,v,plan)
                error=errors(teacher,output[0,sites])
                if error['relative_L2']>.01 or error['max_abs']>.02:raise RuntimeError(f'selected FP32 gate failed: {error}')
                if policy=='split_shared':
                    equivalence=errors(outputs['shared'],output)
                    if equivalence['relative_L2']>.01:raise RuntimeError('split-shared numerical gate failed')
                else:equivalence=None
                outputs[policy]=output
                mask=plan['mask'].cpu()
                if args.reference_routes:
                    reference=torch.load(args.reference_routes/f'{case.name}_{capture["frame"]}_{policy}_route.pt',weights_only=True)
                    if not torch.equal(mask,reference):raise RuntimeError('prototype cache changed the frozen route')
                row=dict(case=case.name,capture_sha256=source_sha,frame=capture['frame'],layer=14,policy=policy,
                    q_shape=list(q.shape),k_shape=list(k.shape),sampled_teacher_Q=len(sites),
                    groups=plan['groups'],eligible_frames=len(eligible),mandatory_frames=len(protected),
                    optional_tokens_per_query_head=plan['budget'],
                    scheduled_pairs=q.shape[1]*q.shape[2]*(len(protected)*880+plan['budget']),
                    physical_union_tokens=int(plan['statistics'][0]),head_union_tokens=int(plan['statistics'][1]),
                    changed_group_head_frames_from_shared=int(plan['statistics'][2]),
                    original_raw_KV_packed_bytes=plan['packed_KV_bytes'],query_pack_bytes=plan['query_pack_bytes'],
                    index_temporary_bytes=plan['route_index_temporary_bytes'],peak_GPU_allocated_bytes=peak,
                    replay_base_GPU_allocated_bytes=base_allocated,peak_minus_base_bytes=peak-base_allocated,
                    memory_scope='replay process includes retained teacher/reference tensors; live peak measured separately',
                    cold_sample=cold_sample,prototype_cache=router.audit(),
                    reference_route_equal=True if args.reference_routes else None,
                    independent_selected_teacher_error=error,split_shared_error=equivalence,
                    independent_dense_output_error=errors(dense,output[0,sites]),
                    samples=times,median={key:statistics.median(x[key] for x in times) for key in times[0]},
                    route_sha256=hashlib.sha256(mask.numpy().tobytes()).hexdigest())
                records.append(row)
                (args.output/f'{case.name}_{capture["frame"]}_{policy}.json').write_text(json.dumps(row,indent=2)+'\n')
                torch.save(mask,args.output/f'{case.name}_{capture["frame"]}_{policy}_route.pt')
                print(json.dumps({key:row[key] for key in ('case','frame','policy','changed_group_head_frames_from_shared','median','independent_dense_output_error')}),flush=True)
            # Existing Block64 fast remains an external granularity/cost control.
            mapping=torch.full((k.shape[1],),-2,dtype=torch.long)
            mapping[capture['protected']]=-1
            for g,ids in enumerate(capture['token_blocks']):mapping[ids]=g
            if (mapping==-2).any():raise RuntimeError('incomplete external fast graph')
            mapping=mapping.cuda();costs=capture['counts'].tolist();budget=sum(costs)//2
            fast_times=[]
            for repeat in range(args.repeats+1):
                torch.cuda.synchronize();began=time.perf_counter()
                a=normalized_values(q[0,sites],km,vm,cnt)
                chosen,coverage,used=select_static_once(a,costs,budget)
                fast,visible=execute_per_head(q,k,v,chosen,mapping,len(capture['protected'])+budget)
                torch.cuda.synchronize()
                if repeat:fast_times.append(time.perf_counter()-began)
            if not torch.equal(chosen.cpu(),capture['selected_group_mask']):raise RuntimeError('old fast route not reproduced')
            records.append(dict(case=case.name,frame=capture['frame'],policy='old_block64_fast',
                complete_wall_samples_s=fast_times,median_complete_wall_s=statistics.median(fast_times),
                independent_dense_output_error=errors(dense,fast[0,sites]),original_route_reproduced=True,
                scheduled_pairs=int(visible.sum())*q.shape[1]))
            del q,k,v,km,vm,cnt,outputs,output,teacher,dense,qq,plan,router,fast,visible,a
    result=dict(status='pass',actual_GPU=torch.cuda.get_device_name(),torch=torch.__version__,records=records,
        implementation_hashes={name:hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
            for name in ('scripts/probe_frame_query_groups.py','adapters/longlive_sparse/frame_query_groups.py')},
        selector_inputs='past clean prototypes and current sampled Q only',new_generated_videos=0,fallback=False,
        timing_scope='warm fixed-input component replay; CUDA stream spans include host submission gaps; not end-to-end video speed',
        quality_scope='independent output diagnostics only; no MSE promotion gate',
        provenance='existing 5B observer, not a fresh generation; resident KV, no archive H2D reduction claim')
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
