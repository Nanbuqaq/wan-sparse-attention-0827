"""Bounded offline capture of actual Attention inputs; never a routing input."""
from collections import Counter
import hashlib
from pathlib import Path
import time
from unittest.mock import patch

import torch

from .native_commit_replay import owned_cpu


def spatial_query_indices(height,width,frames):
    sites=((height//2,width//2),(height//4,width//4),
           (height//4,3*width//4),(3*height//4,width//2))
    if min(height,width)<4 or frames<=0:raise ValueError('unsupported capture geometry')
    return [f*height*width+y*width+x for f in range(frames) for y,x in sites]


class NativeAttentionTeacherCapture:
    def __init__(self,pipe,*,query_frame,token_grid,budget=4*1024**3):
        self.pipe=pipe;self.query_frame=query_frame;self.grid=token_grid;self.budget=budget
        if token_grid[0]*token_grid[1]!=pipe.frame_seq_length or pipe.sampling_steps!=4:
            raise ValueError('capture requires known block geometry and native four-step schedule')
        self.layers=(0,14,29);self.phases=(0,3,4);self.counts=Counter()
        self.active=None;self.layer=0;self.records=[];self.bytes=0;self.capture_wall_s=0.
        self.handle=None;self.attention_patch=None

    def before(self,owner,values,kwargs):
        frame=int(kwargs['current_start'])//self.pipe.frame_seq_length
        phase=self.counts[frame];self.counts[frame]+=1
        self.active=(frame,phase);self.layer=0

    def observe(self,original,q,k,v,*args,**kwargs):
        layer=self.layer;self.layer+=1
        row=None;selected=None
        if self.active and self.active[0]==self.query_frame and self.active[1] in self.phases and layer in self.layers:
            began=time.perf_counter()
            if q.shape[0]!=1 or q.dtype!=torch.bfloat16 or k.dtype!=q.dtype or v.dtype!=q.dtype:
                raise ValueError('teacher capture requires original BF16 batch1 Q/K/V')
            if args or kwargs:raise ValueError('unregistered native Attention mask/scale options')
            frames=q.shape[1]//self.pipe.frame_seq_length
            indices=spatial_query_indices(*self.grid,frames)
            needed=(2*len(indices)*q.shape[2]*q.shape[3]+k.numel()+v.numel())*q.element_size()+len(indices)*8
            if self.bytes+needed>self.budget:raise RuntimeError('offline capture exceeds explicit CPU budget')
            selected=torch.tensor(indices,device=q.device,dtype=torch.long)
            row=dict(query_frame=self.active[0],phase=self.active[1],layer=layer,
                q=owned_cpu(q.index_select(1,selected)),k=owned_cpu(k),v=owned_cpu(v),
                query_indices=torch.tensor(indices,dtype=torch.long),full_Q_tokens=int(q.shape[1]),
                frame_tokens=self.pipe.frame_seq_length,token_grid=list(self.grid),
                query_sites=['center','upper_left','upper_right','lower_center'],
                Q_and_K_already_RoPE_positioned=True,attention_causal_mask=False,softmax_scale=q.shape[-1]**-0.5)
            self.records.append(row);self.bytes+=needed
            self.capture_wall_s+=time.perf_counter()-began
        result=original(q,k,v,*args,**kwargs)
        if row is not None:
            began=time.perf_counter()
            row['native_output']=owned_cpu(result.index_select(1,selected))
            self.capture_wall_s+=time.perf_counter()-began
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

    def export(self,path):
        expected={(self.query_frame,p,l) for p in self.phases for l in self.layers}
        observed={(r['query_frame'],r['phase'],r['layer']) for r in self.records}
        if observed!=expected or len(self.records)!=len(expected):raise RuntimeError('incomplete/duplicate teacher capture grid')
        path=Path(path)
        if path.exists():raise FileExistsError(path)
        torch.save(dict(schema='native_attention_teacher_v1',records=self.records,
            online_routing_may_not_access=True),path)
        with path.open('rb') as handle:sha=hashlib.file_digest(handle,'sha256').hexdigest()
        return dict(path=str(path),sha256=sha,file_bytes=path.stat().st_size(),records=len(self.records),
            CPU_tensor_peak_bytes=self.bytes,CPU_budget_bytes=self.budget,capture_D2H_and_copy_wall_s=self.capture_wall_s,
            capture_wall_includes_input_and_output_readiness_wait=True,
            input_grid=sorted(observed),offline_only_not_read_by_online_method=True,
            method_timing_not_comparable_to_uncaptured_control=True)
