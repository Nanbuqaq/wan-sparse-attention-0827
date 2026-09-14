"""Bitwise raw-KV/Q gather and output scatter for one original FA2 call.

Does not change any selected frame or reuse current noisy data. Routes must be
valid fixed-size query partitions; FrameQueryRouter owns that geometry contract.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _pack_kv(K,V,F,OK,OV,FT:tl.constexpr,H:tl.constexpr,D:tl.constexpr,NF:tl.constexpr,B:tl.constexpr):
    gh=tl.program_id(0)
    t=tl.program_id(1)*B+tl.arange(0,B)
    d=tl.arange(0,D)
    f=tl.load(F+gh*NF+t//FT,mask=t<NF*FT,other=0)
    source=((f*FT+t%FT)*H+gh%H)[:,None]*D+d[None,:]
    dest=(gh*(NF*FT)+t)[:,None]*D+d[None,:]
    kv=tl.load(K+source,mask=(t<NF*FT)[:,None],other=0)
    vv=tl.load(V+source,mask=(t<NF*FT)[:,None],other=0)
    tl.store(OK+dest,kv,mask=(t<NF*FT)[:,None])
    tl.store(OV+dest,vv,mask=(t<NF*FT)[:,None])


@triton.jit
def _pack_q(Q,I,O,H:tl.constexpr,D:tl.constexpr,L:tl.constexpr,B:tl.constexpr):
    gh=tl.program_id(0)
    t=tl.program_id(1)*B+tl.arange(0,B)
    d=tl.arange(0,D)
    qi=tl.load(I+(gh//H)*L+t,mask=t<L,other=0)
    src=(qi*H+gh%H)[:,None]*D+d[None,:]
    dst=(gh*L+t)[:,None]*D+d[None,:]
    x=tl.load(Q+src,mask=(t<L)[:,None],other=0)
    tl.store(O+dst,x,mask=(t<L)[:,None])


@triton.jit
def _scatter_output(X,I,O,H:tl.constexpr,D:tl.constexpr,L:tl.constexpr,B:tl.constexpr):
    gh=tl.program_id(0)
    t=tl.program_id(1)*B+tl.arange(0,B)
    d=tl.arange(0,D)
    qi=tl.load(I+(gh//H)*L+t,mask=t<L,other=0)
    src=(gh*L+t)[:,None]*D+d[None,:]
    dst=(qi*H+gh%H)[:,None]*D+d[None,:]
    x=tl.load(X+src,mask=(t<L)[:,None],other=0)
    tl.store(O+dst,x,mask=(t<L)[:,None])


def pack_frame_routes(q,k,v,plan):
    frames,meta=plan['frame_ids'],plan['meta']
    groups,heads,nframes=frames.shape
    length=meta['query_ids'].shape[1];dim=q.shape[-1];ft=meta['offsets'].numel()
    if q.shape[0]!=1 or q.dtype!=torch.bfloat16 or k.shape!=v.shape or dim!=128:
        raise ValueError('qualified single BF16 DiT window only')
    if not all(x.is_contiguous() for x in (q,k,v,frames,meta['query_ids'])):
        raise ValueError('explicit contiguous input required; no hidden copy/fallback')
    pk=torch.empty((groups*heads*nframes*ft,1,dim),device=q.device,dtype=q.dtype)
    pv=torch.empty_like(pk)
    pq=torch.empty((groups*heads*length,1,dim),device=q.device,dtype=q.dtype)
    _pack_kv[(groups*heads,triton.cdiv(nframes*ft,16))](k,v,frames,pk,pv,ft,heads,dim,nframes,16,num_warps=4)
    _pack_q[(groups*heads,triton.cdiv(length,16))](q,meta['query_ids'],pq,heads,dim,length,16,num_warps=4)
    return pq,pk,pv


def execute_fused_frame_routes(q,k,v,plan,timing=None):
    from flash_attn import flash_attn_varlen_func
    frames,meta=plan['frame_ids'],plan['meta']
    groups,heads,nframes=frames.shape
    length=meta['query_ids'].shape[1];dim=q.shape[-1];ft=meta['offsets'].numel()
    if timing is not None:timing[0].record()
    pq,pk,pv=pack_frame_routes(q,k,v,plan)
    if timing is not None:timing[1].record()
    out=flash_attn_varlen_func(pq,pk,pv,meta['cuq'],meta['cuk'],length,nframes*ft,dropout_p=0.,causal=False)
    if timing is not None:timing[2].record()
    result=torch.empty_like(q)
    _scatter_output[(groups*heads,triton.cdiv(length,16))](out,meta['query_ids'],result,heads,dim,length,16,num_warps=4)
    if timing is not None:timing[3].record()
    return result
