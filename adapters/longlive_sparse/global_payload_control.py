"""Read-only global K/V controls at a returned phase, with fixed token count."""
import time
import torch
from .immutable_source_reader import ImmutableSourceReader
from .layer_stream_source_reader import LayerStreamSourceReader


def global_read_buffers(k,v,global_tokens,mode):
    if mode not in ('original','zero_v','zero_kv') or k.shape!=v.shape or not 0<global_tokens<k.shape[1]:
        raise ValueError('valid global prefix and matching K/V required')
    if mode=='original':return k,v,0
    changed_v=v.clone();changed_v[:,:global_tokens].zero_()
    changed_k=k
    if mode=='zero_kv':changed_k=k.clone();changed_k[:,:global_tokens].zero_()
    owned=changed_v.numel()*changed_v.element_size()
    if mode=='zero_kv':owned+=changed_k.numel()*changed_k.element_size()
    return changed_k,changed_v,owned


class GlobalPayloadMixin:
    def __init__(self,*args,global_payload,**kwargs):
        super().__init__(*args,**kwargs)
        if (global_payload not in ('original','zero_v','zero_kv') or self.context_policy!='anchor_transition'
                or self.source_order!='after_global' or self.source_policy!='full_once'
                or self.source_stage_policy!='all' or self.snapshot_window!='latest8'):
            raise ValueError('global payload study keeps the exact original global/source/current graph')
        self.global_payload=global_payload;self.global_events=[];self.global_peak_buffer_bytes=0
        self.global_prepare_host_s=0.;self.global_calls=0

    def dispatch(self,layer,original,q,k,v,**kwargs):
        active=self.return_phase is not None and self.return_phase==self.phase
        if not active or self.global_payload=='original':return super().dispatch(layer,original,q,k,v,**kwargs)
        g=kwargs['global_sink_tokens'];owned=0;host_s=0.;events=None
        def changed(qq,kk,vv):
            nonlocal owned,host_s,events
            began=time.perf_counter();events=[torch.cuda.Event(enable_timing=True) for _ in range(2)];events[0].record()
            rk,rv,owned=global_read_buffers(kk,vv,g,self.global_payload)
            events[1].record();host_s=time.perf_counter()-began
            return original(qq,rk,rv)
        out=super().dispatch(layer,changed,q,k,v,**kwargs)
        row=self.rows[-1];row.update(global_payload=self.global_payload,global_read_tokens=g,
            global_read_buffer_bytes=owned,global_read_prepare_host_s=host_s,stored_native_global_mutated=False)
        self.global_calls+=1;self.global_prepare_host_s+=host_s;self.global_peak_buffer_bytes=max(self.global_peak_buffer_bytes,owned)
        self.global_events.append((row,events))
        if len(self.global_events)>2048:raise RuntimeError('global read diagnostic event bound exceeded')
        return out

    def audit(self):
        result=super().audit()
        if self.global_events:torch.cuda.synchronize()
        for row,events in self.global_events:row['global_read_prepare_stream_ms']=events[0].elapsed_time(events[1])
        result['global_payload_control']=dict(mode=self.global_payload,calls=self.global_calls,
            extra_read_buffer_peak_bytes=self.global_peak_buffer_bytes,prepare_host_s=self.global_prepare_host_s,
            native_cache_unchanged=True,token_count_and_positions_unchanged=True,
            no_storage_saving_claim=True,timing_scope='read preparation lies inside parent attention span; do not add them')
        return result


class GlobalPayloadResidentReader(GlobalPayloadMixin,ImmutableSourceReader):pass
class GlobalPayloadStreamReader(GlobalPayloadMixin,LayerStreamSourceReader):pass
