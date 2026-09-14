"""Equal-kernel/control probe of grid order on actual 5B frame routes."""
import argparse,json,statistics,sys
from pathlib import Path
import torch,triton
import triton.language as tl
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adapters.longlive_sparse.frame_query_groups import FrameQueryRouter
from adapters.longlive_sparse.fused_frame_routes import pack_frame_routes

@triton.jit
def pack(K,V,F,OK,OV,FT:tl.constexpr,H:tl.constexpr,D:tl.constexpr,NF:tl.constexpr,HEAD_ADJ:tl.constexpr,B:tl.constexpr):
    gh=tl.program_id(0) if HEAD_ADJ else tl.program_id(1)
    block=tl.program_id(1) if HEAD_ADJ else tl.program_id(0)
    t=block*B+tl.arange(0,B);d=tl.arange(0,D)
    f=tl.load(F+gh*NF+t//FT,mask=t<NF*FT,other=0)
    src=((f*FT+t%FT)*H+gh%H)[:,None]*D+d[None,:]
    dst=(gh*(NF*FT)+t)[:,None]*D+d[None,:]
    a=tl.load(K+src,mask=(t<NF*FT)[:,None],other=0)
    b=tl.load(V+src,mask=(t<NF*FT)[:,None],other=0)
    tl.store(OK+dst,a,mask=(t<NF*FT)[:,None]);tl.store(OV+dst,b,mask=(t<NF*FT)[:,None])

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    rows=[]
    for case in a.cases:
        data=torch.load(case/'steady_observer.pt',map_location='cpu',weights_only=True,mmap=True)
        for c in data['calls']:
            q,k,v=[c[x].cuda() for x in ['q','k','v']]
            km,vm,cnt=[c[x].cuda() for x in ['key_mean','value_mean','counts']]
            eligible=[i for i,_ in c['eligible']];protected=sorted({i//880 for i in c['protected']})
            summaries=[(km[i*14:(i+1)*14],vm[i*14:(i+1)*14],cnt[i*14:(i+1)*14]) for i in range(len(eligible))]
            for policy in ['shared','split_shared','split_specific']:
                router=FrameQueryRouter(policy,c['token_grid']);plan=router.prepare(q,k,eligible,protected,summaries,880)
                pq,rk,rv=pack_frame_routes(q,k,v,plan);del pq
                ok,ov=torch.empty_like(rk),torch.empty_like(rv)
                g,h,nf=plan['frame_ids'].shape;tiles=triton.cdiv(nf*880,16);times={True:[],False:[]}
                for rep in range(9):
                    for adjacent in ([True,False] if rep%2==0 else [False,True]):
                        grid=(g*h,tiles) if adjacent else (tiles,g*h)
                        torch.cuda.synchronize();start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)];start.record()
                        pack[grid](k,v,plan['frame_ids'],ok,ov,880,h,128,nf,adjacent,16,num_warps=4)
                        end.record();end.synchronize()
                        assert torch.equal(ok,rk) and torch.equal(ov,rv)
                        if rep>=2:times[adjacent].append(start.elapsed_time(end))
                row=dict(case=case.name,frame=c['frame'],policy=policy,packed_KV_bytes=2*rk.numel()*rk.element_size(),
                    same_output_bitwise=True,heads_adjacent_ms=statistics.median(times[True]),tokens_adjacent_ms=statistics.median(times[False]),samples={str(k):v for k,v in times.items()})
                rows.append(row);print(json.dumps(row),flush=True);del rk,rv,ok,ov
    with (a.output/'result.json').open('x') as f:json.dump(dict(status='pass',GPU=torch.cuda.get_device_name(),rows=rows,
        scope='same arithmetic, same read graph, same output layout; only program-ID mapping changes; no cache-transaction/SM-utilization or full-video claim'),f,indent=2)

if __name__=='__main__':main()
