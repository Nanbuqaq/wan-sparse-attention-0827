"""Offline all-layer sampled role statistics with bounded per-head FP32 scratch.

Returns native Attention unchanged. Complete K/V is read ONLY by this isolated
observer; stored statistics support group reweight/deletion, not arbitrary new
token routing or temporal-K replay. No observer tensor is an online selector input.
"""
from collections import Counter
import hashlib
from pathlib import Path
import time
from unittest.mock import patch

import torch

from .native_attention_teacher import spatial_query_indices


@torch.inference_mode()
def sampled_role_statistics(q,k,v,group_width):
    if q.ndim!=4 or q.shape[0]!=1 or k.shape!=v.shape or k.shape[1]!=4*group_width:
        raise ValueError('batch1 with four full role groups required')
    previous_tf32=torch.backends.cuda.matmul.allow_tf32
    if q.device.type=='cuda':torch.backends.cuda.matmul.allow_tf32=False
    zs=[];outputs=[]
    try:
        for head in range(q.shape[2]):
            query=q[0,:,head].float().contiguous();head_z=[];head_output=[]
            for group in range(4):
                selection=slice(group*group_width,(group+1)*group_width)
                key=k[0,selection,head].float().contiguous();value=v[0,selection,head].float().contiguous()
                score=(query@key.T)*(q.shape[-1]**-.5)
                head_z.append(torch.logsumexp(score,dim=-1))
                head_output.append(score.softmax(-1)@value)
            zs.append(torch.stack(head_z,dim=-1));outputs.append(torch.stack(head_output,dim=-2))
        # CPU copies also complete this observer's device work before precision restoration.
        return torch.stack(zs).cpu(),torch.stack(outputs).cpu()
    finally:
        if q.device.type=='cuda':torch.backends.cuda.matmul.allow_tf32=previous_tf32


class NativeLayerRoleProbe:
    def __init__(self,pipe,*,token_grid,budget=512*1024**2):
        if pipe.sampling_steps!=4 or pipe.frame_seq_length!=token_grid[0]*token_grid[1] or pipe.local_attn_size!=32:
            raise ValueError('qualified native four-step/local32 geometry required')
        if pipe.use_relative_rope or pipe.guidance_scale!=1 or pipe.quantize_kv:
            raise ValueError('qualified absolute BF16 CFG1 required')
        self.pipe=pipe;self.grid=token_grid;self.budget=budget;self.bytes=0;self.records=[]
        self.frames=(96,120);self.phases=(0,3,4);self.layers=tuple(range(30));self.counts=Counter()
        self.active=None;self.layer=0;self.handle=None;self.attention_patch=None;self.observer_wall_s=0.

    def before(self,owner,values,kwargs):
        frame=int(kwargs['current_start'])//self.pipe.frame_seq_length
        self.active=(frame,self.counts[frame]);self.counts[frame]+=1;self.layer=0

    def observe(self,original,q,k,v,*args,**kwargs):
        layer=self.layer;self.layer+=1
        result=original(q,k,v,*args,**kwargs)
        if not (self.active and self.active[0] in self.frames and self.active[1] in self.phases and layer in self.layers):return result
        started=time.perf_counter()
        if args or kwargs or q.dtype!=torch.bfloat16 or k.dtype!=q.dtype or v.dtype!=q.dtype or q.shape[0]!=1:
            raise ValueError('unregistered attention semantics/dtype')
        frames=q.shape[1]//self.pipe.frame_seq_length
        if frames!=8 or k.shape[1]!=32*self.pipe.frame_seq_length:raise ValueError('role accounting requires full native32')
        indices=spatial_query_indices(*self.grid,frames)
        needed=len(indices)*q.shape[2]*(4+4*q.shape[-1])*4+len(indices)*q.shape[2]*q.shape[-1]*2+len(indices)*8
        if self.bytes+needed>self.budget:raise RuntimeError('compact offline role budget exceeded')
        chosen=torch.tensor(indices,device=q.device,dtype=torch.long)
        log_z,role_outputs=sampled_role_statistics(q.index_select(1,chosen),k,v,8*self.pipe.frame_seq_length)
        native=result.index_select(1,chosen).detach().cpu()[0].permute(1,0,2).contiguous()
        labels=(['initial','source','away','current'] if self.active[0]==96 else
                ['initial','first_return_anchor','recent_return','current'])
        self.records.append(dict(query_frame=self.active[0],phase=self.active[1],layer=layer,labels=labels,
            log_z=log_z,role_outputs=role_outputs,native_output=native,query_indices=torch.tensor(indices),
            sampled_queries_per_head=len(indices),heads=q.shape[2],head_dim=q.shape[3],
            full_Q_tokens=q.shape[1],K_tokens=k.shape[1],frame_tokens=self.pipe.frame_seq_length,
            query_sites=['center','upper_left','upper_right','lower_center'],FP32_reference_TF32_disabled=True))
        self.bytes+=needed;self.observer_wall_s+=time.perf_counter()-started
        return result

    def attach(self):
        import wan_5b.modules.causal_model as native
        self.handle=self.pipe.generator.register_forward_pre_hook(self.before,with_kwargs=True)
        original=native.attention
        self.attention_patch=patch.object(native,'attention',lambda q,k,v,*a,**kw:self.observe(original,q,k,v,*a,**kw))
        self.attention_patch.start()

    def detach(self):
        if self.handle is not None:self.handle.remove();self.handle=None
        if self.attention_patch is not None:self.attention_patch.stop();self.attention_patch=None

    def export(self,path,*,allow_partial=False):
        expected={(f,p,l) for f in self.frames for p in self.phases for l in self.layers}
        actual={(r['query_frame'],r['phase'],r['layer']) for r in self.records}
        complete=actual==expected and len(self.records)==len(expected)
        if not complete and not allow_partial:raise RuntimeError('incomplete/duplicate all-layer probe')
        path=Path(path)
        if path.exists():raise FileExistsError(path)
        torch.save(dict(schema='native_layer_role_probe_v1',records=self.records,
            offline_teacher_only=True,online_routing_may_not_access=True,full_QKV_not_stored=True,complete=complete),path)
        with path.open('rb') as handle:sha=hashlib.file_digest(handle,'sha256').hexdigest()
        return dict(path=str(path),sha256=sha,records=len(self.records),complete=complete,CPU_tensor_bytes=self.bytes,CPU_budget=self.budget,
            file_bytes=path.stat().st_size,observer_wall_s=self.observer_wall_s,
            only_sampled_group_statistics_not_full_QKV=True,method_timing_not_comparable=True,
            all_layer_profile_does_not_freeze_an_online_policy=True)
