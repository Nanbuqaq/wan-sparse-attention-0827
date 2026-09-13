"""Bounded KV-major reference, retaining only R key-tile partials at a time."""
import torch
import triton
import triton.language as tl
from .dataflow_reference import _visibility


@triton.jit
def partial_window(QP,KP,VP,TAG,PART,LSE,Q:tl.constexpr,K:tl.constexpr,D:tl.constexpr,
                   REUSE:tl.constexpr,R:tl.constexpr,START,BQ:tl.constexpr=64,BK:tl.constexpr=64):
    h,kb=tl.program_id(0),tl.program_id(1)
    ki=(START+kb)*BK+tl.arange(0,BK);di=tl.arange(0,D)
    k=tl.load(KP+h*K*D+ki[:,None]*D+di[None,:],ki[:,None]<K,0)
    v=tl.load(VP+h*K*D+ki[:,None]*D+di[None,:],ki[:,None]<K,0)
    tags=tl.load(TAG+ki,ki<K,-1)
    for qb in range(tl.cdiv(Q,BQ)):
        qi=qb*BQ+tl.arange(0,BQ)
        allowed=_visibility(qi,ki,tags,Q,K,REUSE)
        lse=tl.full((BQ,),float('-inf'),tl.float32)
        if tl.sum(allowed.to(tl.int32))>0:
            q=tl.load(QP+h*Q*D+qi[:,None]*D+di[None,:],qi[:,None]<Q,0)
            scores=tl.dot(q,tl.trans(k))*(1.4426950408889634/(D**.5))
            scores=tl.where(allowed,scores,float('-inf'))
            m=tl.max(scores,1);safe=tl.where(m==float('-inf'),0.,m)
            p=tl.exp2(scores-safe[:,None]);den=tl.sum(p,1)
            out=tl.dot(p.to(v.dtype),v)/tl.maximum(den[:,None],1.e-20)
            lse=tl.where(den>0,m+tl.log2(den),float('-inf'))
            tl.store(PART+((h*R+kb)*Q+qi[:,None])*D+di[None,:],out,(qi[:,None]<Q)&(den[:,None]>0))
        tl.store(LSE+(h*R+kb)*Q+qi,lse,qi<Q)


@triton.jit
def merge_window(PART,LSE,ACC,MAXIMUM,DENOM,OUT,Q:tl.constexpr,D:tl.constexpr,
                 R:tl.constexpr,FIRST:tl.constexpr,LAST:tl.constexpr,BQ:tl.constexpr=16):
    h,qb=tl.program_id(0),tl.program_id(1);qi=qb*BQ+tl.arange(0,BQ);di=tl.arange(0,D)
    if FIRST:
        m=tl.full((BQ,),float('-inf'),tl.float32);den=tl.zeros((BQ,),tl.float32);acc=tl.zeros((BQ,D),tl.float32)
    else:
        m=tl.load(MAXIMUM+h*Q+qi,qi<Q,float('-inf'));den=tl.load(DENOM+h*Q+qi,qi<Q,0)
        acc=tl.load(ACC+h*Q*D+qi[:,None]*D+di[None,:],qi[:,None]<Q,0)
    for b in range(R):
        score=tl.load(LSE+(h*R+b)*Q+qi,qi<Q,float('-inf'))
        new_m=tl.maximum(m,score);safe=tl.where(new_m==float('-inf'),0.,new_m)
        alpha=tl.exp2(m-safe);beta=tl.exp2(score-safe)
        part=tl.load(PART+((h*R+b)*Q+qi[:,None])*D+di[None,:],(qi[:,None]<Q)&(score[:,None]!=float('-inf')),0.)
        acc=acc*alpha[:,None]+part*beta[:,None];den=den*alpha+beta;m=new_m
    if LAST:
        tl.store(OUT+h*Q*D+qi[:,None]*D+di[None,:],acc/tl.maximum(den[:,None],1.e-20),qi[:,None]<Q)
    else:
        tl.store(ACC+h*Q*D+qi[:,None]*D+di[None,:],acc,qi[:,None]<Q)
        tl.store(MAXIMUM+h*Q+qi,m,qi<Q);tl.store(DENOM+h*Q+qi,den,qi<Q)


class BoundedKVOut:
    def __init__(self,inputs,tiles=4):
        inputs.validate()
        if tiles not in (4,8):raise ValueError('registered R is4 or8')
        self.inputs=inputs;self.tiles=tiles;h,q,d=inputs.query.shape;device=inputs.query.device
        self.partial=torch.empty((h,tiles,q,d),device=device,dtype=torch.float32)
        self.lse=torch.empty((h,tiles,q),device=device,dtype=torch.float32)
        self.acc=torch.empty((h,q,d),device=device,dtype=torch.float32)
        self.maximum=torch.empty((h,q),device=device,dtype=torch.float32);self.denom=torch.empty_like(self.maximum)
        self.output=torch.empty_like(inputs.query)

    @property
    def workspace_bytes(self):
        return sum(t.numel()*t.element_size() for t in (self.partial,self.lse,self.acc,self.maximum,self.denom))

    def run(self):
        c=self.inputs;h,q,d=c.query.shape;k=c.key.shape[1];nt=triton.cdiv(k,64)
        for start in range(0,nt,self.tiles):
            partial_window[(h,self.tiles)](c.query,c.key,c.value,c.key_roles,self.partial,self.lse,
                q,k,d,c.reuse,self.tiles,start,num_warps=4,num_stages=1)
            merge_window[(h,triton.cdiv(q,16))](self.partial,self.lse,self.acc,self.maximum,self.denom,self.output,
                q,d,self.tiles,start==0,start+self.tiles>=nt,num_warps=4,num_stages=1)
        return self.output
