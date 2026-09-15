"""Value-only temporal/spatial grouping with original restored source keys."""
import time
import torch
from .immutable_source_reader import ImmutableSourceReader
from .layer_stream_source_reader import LayerStreamSourceReader


def summarize_source_values(v,grid,kind):
    ht,wt=grid;b,n,h,d=v.shape
    if kind not in ('temporal8','spatial2x4') or n!=8*ht*wt or ht%2 or wt%4:
        raise ValueError('registered source8 and divisible spatial grid required')
    if kind=='temporal8':
        return v.reshape(b,8,ht,wt,h,d).mean(1,dtype=torch.float32).to(v.dtype).reshape(b,ht*wt,h,d)
    return v.reshape(b,8,ht//2,2,wt//4,4,h,d).mean((3,5),dtype=torch.float32).to(v.dtype).reshape(b,ht*wt,h,d)


def fill_grouped_source_values(destination,summary,grid,kind):
    ht,wt=grid;b,n,h,d=destination.shape
    if n!=8*ht*wt or summary.shape!=(b,ht*wt,h,d):raise ValueError('source/prototype geometry differs')
    if kind=='temporal8':destination.reshape(b,8,ht,wt,h,d).copy_(summary.reshape(b,1,ht,wt,h,d))
    elif kind=='spatial2x4':destination.reshape(b,8,ht//2,2,wt//4,4,h,d).copy_(summary.reshape(b,8,ht//2,1,wt//4,1,h,d))
    else:raise ValueError('unknown source value grouping')


class SourceValueGroupsMixin:
    def __init__(self,*args,source_value_groups,**kwargs):
        super().__init__(*args,**kwargs)
        if (source_value_groups not in ('temporal8','spatial2x4') or self.source_policy!='full_once'
            or self.source_stage_policy!='all' or self.context_policy!='anchor_transition'
            or self.source_order!='after_global' or self.snapshot_window!='latest8'):
            raise ValueError('value groups fix the original latest8 source graph')
        self.value_groups=source_value_groups;self.value_group_binding=None;self.value_prototypes={}
        self.value_group_events=[];self.value_group_calls=0;self.value_prototype_peak_bytes=0;self.value_read_peak_bytes=0

    def before(self,owner,values,kwargs):
        super().before(owner,values,kwargs)
        binding=self.active_side['binding_sha'] if self.active_side is not None else None
        if binding!=self.value_group_binding:self.value_prototypes.clear();self.value_group_binding=binding

    def dispatch(self,layer,original,q,k,v,**kwargs):
        frame=self.active_start//self.frame_tokens
        if not self.source_is_allowed(layer,frame):return super().dispatch(layer,original,q,k,v,**kwargs)
        def projected(qq,kk,vv):
            began=time.perf_counter();events=[torch.cuda.Event(enable_timing=True) for _ in range(2)];events[0].record()
            g=kwargs['global_sink_tokens'];n=8*self.frame_tokens;raw=vv[:,g:g+n]
            summary=self.value_prototypes.get(layer)
            if summary is None:
                summary=summarize_source_values(raw,self.token_grid,self.value_groups);self.value_prototypes[layer]=summary
                self.value_prototype_peak_bytes=max(self.value_prototype_peak_bytes,sum(t.numel()*t.element_size() for t in self.value_prototypes.values()))
            values=vv.clone();fill_grouped_source_values(values[:,g:g+n],summary,self.token_grid,self.value_groups)
            events[1].record();self.value_group_events.append((events,time.perf_counter()-began))
            self.value_read_peak_bytes=max(self.value_read_peak_bytes,values.numel()*values.element_size());self.value_group_calls+=1
            return original(qq,kk,values)
        return super().dispatch(layer,projected,q,k,v,**kwargs)

    def audit(self):
        result=super().audit()
        if self.value_group_events:torch.cuda.synchronize()
        result['source_value_groups']=dict(kind=self.value_groups,modified_calls=self.value_group_calls,
            prototype_GPU_peak_bytes=self.value_prototype_peak_bytes,extra_read_buffer_peak_bytes=self.value_read_peak_bytes,
            prepare_host_s=sum(t for _,t in self.value_group_events),prepare_stream_ms=sum(e[0].elapsed_time(e[1]) for e,_ in self.value_group_events),
            original_source_K_unmodified=True,global_and_native_V_unmodified=True,
            logical_value_prototype_tokens_per_layer=self.frame_tokens,original_source_tokens_per_layer=8*self.frame_tokens,
            prototype_creation='first admitted source read; committed immutable V only; no query or future outputs',
            original_CPU_archive_and_H2D_still_retained=True,physical_storage_saving_claimed=False,
            timing_scope='projection is inside parent attention span; not additive')
        return result


class SourceValueGroupsResidentReader(SourceValueGroupsMixin,ImmutableSourceReader):pass
class SourceValueGroupsStreamReader(SourceValueGroupsMixin,LayerStreamSourceReader):pass
