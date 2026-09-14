"""Frozen real 5B read graphs: shared mandatory versus repeated grouped packing."""
import argparse,hashlib,json,statistics,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adapters.longlive_sparse.frame_query_groups import FrameQueryRouter
from adapters.longlive_sparse.shared_mandatory_attention import execute_shared_mandatory_routes
from probe_frame_query_groups import errors, selected_teacher

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=7)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('real GPU required')
    rows=[]
    for case in args.cases:
        path=case/'steady_observer.pt';capture=torch.load(path,weights_only=True,map_location='cpu',mmap=True)
        with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        for c in capture['calls']:
            if c['frame'] not in (24,88):continue
            q,k,v=[c[x].cuda() for x in ('q','k','v')]
            km,vm,cnt=[c[x].cuda() for x in ('key_mean','value_mean','counts')]
            eligible=[i for i,owner in c['eligible']]
            assert all(owner[0]=='native' and owner[1]<c['frame'] for _,owner in c['eligible'])
            protected=sorted({i//880 for i in c['protected']})
            summaries=[(km[i*14:(i+1)*14],vm[i*14:(i+1)*14],cnt[i*14:(i+1)*14]) for i in range(len(eligible))]
            router=FrameQueryRouter('split_specific',c['token_grid'])
            plan=router.prepare(q,k,eligible,protected,summaries,880)
            ref=router.execute(q,k,v,plan)
            actual=execute_shared_mandatory_routes(q,k,v,plan)
            teacher=selected_teacher(q,k,v,plan)
            gate=errors(teacher,actual[0,plan['meta']['sites']]);same=errors(ref,actual)
            if gate['relative_L2']>.01 or gate['max_abs']>.02 or same['relative_L2']>.01:
                raise RuntimeError(f'independent selected graph numerical gate failed: {gate}, {same}')
            del teacher,actual,ref
            samples={x:[] for x in ('repeated_group_pack','shared_mandatory_pack')}
            peaks={x:[] for x in samples}
            for repeat in range(args.repeats+2):
                order=list(samples)
                if repeat%2:order.reverse()
                for method in order:
                    torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                    ev=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
                    began=time.perf_counter()
                    out=router.execute(q,k,v,plan,timing=ev) if method=='repeated_group_pack' else execute_shared_mandatory_routes(q,k,v,plan,timing=ev)
                    torch.cuda.synchronize()
                    record=dict(wall_ms=1000*(time.perf_counter()-began),pack_ms=ev[0].elapsed_time(ev[1]),
                        attention_ms=ev[1].elapsed_time(ev[2]),restore_merge_ms=ev[2].elapsed_time(ev[3]))
                    if repeat>=2:
                        samples[method].append(record);peaks[method].append(torch.cuda.max_memory_allocated()-base)
                    del out
            groups,heads,frames=plan['frame_ids'].shape
            original=2*groups*heads*frames*880*q.shape[-1]*q.element_size()
            shared=2*heads*(len(protected)+groups*(frames-len(protected)))*880*q.shape[-1]*q.element_size()
            row=dict(case=case.name,capture_sha256=sha,frame=c['frame'],layer=c['layer'],
                selected_teacher_error=gate,same_graph_reference_error=same,samples=samples,
                median={m:{key:statistics.median(x[key] for x in ss) for key in ss[0]} for m,ss in samples.items()},
                peak_extra_allocated_bytes={m:max(v) for m,v in peaks.items()},
                packed_KV_bytes=dict(repeated=original,shared_mandatory=shared),
                actual_attention_pairs=q.shape[1]*q.shape[2]*frames*880,
                no_archive_H2D=True,selection_and_prototype_cost_common_excluded=True,
                online_backend_pack_includes_partition_construction=True)
            rows.append(row);print(json.dumps(row),flush=True)
    result=dict(status='pass',GPU=torch.cuda.get_device_name(),torch=torch.__version__,rows=rows,
        scope='same real graph GPU replay, not video quality or full delivery speedup; partition merge numerical rather than bitwise equivalence',
        frozen_graph_does_not_reuse_current_noisy_KV_across_steps=True)
    with (args.output/'result.json').open('x') as f:json.dump(result,f,indent=2)

if __name__=='__main__':main()
