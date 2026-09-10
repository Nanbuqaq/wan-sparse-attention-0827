"""Eight head-local exact Q/K/V witnesses for the four existing numeric failures."""
import hashlib
from pathlib import Path
from unittest.mock import patch
import torch
from .native_attention_teacher import spatial_query_indices


TARGETS={(96,0,29):(7,6),(96,4,4):(19,18),(120,0,29):(13,12),(120,3,29):(7,6)}


class NativeNumericWitness:
    def __init__(self,pipe,token_grid):
        self.pipe=pipe;self.grid=token_grid;self.counts={};self.active=None;self.layer=0
        self.records=[];self.handle=None;self.patch=None;self.bytes=0
    def before(self,owner,values,kwargs):
        frame=int(kwargs['current_start'])//self.pipe.frame_seq_length
        phase=self.counts.get(frame,0);self.counts[frame]=phase+1
        self.active=(frame,phase);self.layer=0
    def observe(self,original,q,k,v,*a,**kw):
        layer=self.layer;self.layer+=1
        result=original(q,k,v,*a,**kw)
        key=(*self.active,layer)
        if key not in TARGETS:return result
        if a or kw or q.shape[0]!=1 or q.dtype!=torch.bfloat16 or k.shape!=v.shape:
            raise ValueError('unregistered witness Attention semantics')
        if q.shape[1]!=8*self.pipe.frame_seq_length or k.shape[1]!=32*self.pipe.frame_seq_length:
            raise ValueError('witness must match the original full native32 observation')
        indices=spatial_query_indices(*self.grid,8)
        for head in TARGETS[key]:
            tensors={name:t[0,:,head].detach().contiguous().cpu().clone()
                     for name,t in [('q',q),('k',k),('v',v),('native_output',result)]}
            self.bytes+=sum(t.numel()*t.element_size() for t in tensors.values())
            if self.bytes>256*1024**2:raise RuntimeError('minimal witness exceeds frozen256MiB cap')
            self.records.append(dict(query_frame=key[0],phase=key[1],layer=layer,head=head,
                query_indices=indices,frame_tokens=self.pipe.frame_seq_length,**tensors))
        return result
    def attach(self):
        import wan_5b.modules.causal_model as native
        self.handle=self.pipe.generator.register_forward_pre_hook(self.before,with_kwargs=True)
        original=native.attention
        self.patch=patch.object(native,'attention',lambda q,k,v,*a,**kw:self.observe(original,q,k,v,*a,**kw))
        self.patch.start()
    def detach(self):
        if self.patch is not None:self.patch.stop();self.patch=None
        if self.handle is not None:self.handle.remove();self.handle=None
    def export(self,path,allow_partial=False):
        path=Path(path)
        keys={(r['query_frame'],r['phase'],r['layer'],r['head']) for r in self.records}
        expected={(*k,h) for k,heads in TARGETS.items() for h in heads}
        complete=len(self.records)==8 and keys==expected
        if not complete and not allow_partial:raise RuntimeError('incomplete or duplicate witness')
        if path.exists():raise FileExistsError(path)
        torch.save(dict(schema='native_numeric_witness_v1',records=self.records,complete=complete,
            offline_teacher_only=True,online_routing_may_not_access=True),path)
        with path.open('rb') as h:digest=hashlib.file_digest(h,'sha256').hexdigest()
        return dict(complete=complete,records=len(self.records),CPU_tensor_bytes=self.bytes,
                    sha256=digest,offline_only=True,original_threshold_unchanged=True,
                    selected_for_numeric_diagnosis_not_layer_policy=True)
