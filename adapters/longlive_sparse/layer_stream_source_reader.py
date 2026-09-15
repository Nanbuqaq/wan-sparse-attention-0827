"""Capacity control: one layer of an immutable CPU source at a time.

Same source, binding and single FA2 graph as the resident side bank. Increased
H2D/rephase work is explicitly charged; this is not a latency optimization.
"""
import hashlib
import json
import time
import torch
from .immutable_source_reader import ImmutableSourceReader,source_allowed,request_key
from .native_retimed_memory import binding_coordinates
from .native_temporal_rephase import rephase_temporal_keys


class LayerStreamSourceReader(ImmutableSourceReader):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.source_backend!='concat':raise ValueError('layer stream retains one exact concat FA2 graph')
        self.layer_stream_calls=0;self.layer_stream_prior_wait_s=0.
        self.layer_stream_copy_ms=0.;self.layer_stream_rephase_ms=0.
        self.layer_stream_temporary_peak_bytes=0

    def copy_source_pair(self, cpu_pair, device):
        return tuple(t.to(device=device, copy=True) for t in cpu_pair)

    def _load_side(self,bank,frame,text):
        d=bank['descriptor'];start=d.source_end-8
        coords=binding_coordinates(source_frame=start,target_frame=frame,frames=8,
            source_phase=d.source_phase,current_phase=self.phase,policy='recent_virtual')
        residency=[sum(o is not None and o[0]=='native' and start<=o[1]<d.source_end
                       and o[3]==d.source_phase for o in owners) for owners in self.owners]
        if any(residency):raise RuntimeError('source has not left the full native cache')
        binding=dict(archive_version=d.archive_version,source_frames=list(range(start,d.source_end)),
            source_phase=d.source_phase,admitted_frame=frame,**coords)
        sha=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
        logical=sum(t.numel()*t.element_size() for pair in bank['kv'] for t in pair)
        # Only a shallow list of existing CPU tensor references, never a second
        # raw archive. Base expiration drops it before the next phase archives.
        self.active_side=dict(bank=list(bank['kv']),start=frame,phase=self.phase,request_key=request_key(text),
            binding=binding,binding_sha=sha,versions=[None]*len(bank['kv']),bytes=0)
        self.side_events.append(dict(**binding,binding_sha=sha,pre_read_source_frames_per_native_layer=residency,
            source_absent_entire_native_cache=True,H2D_KV_bytes=0,GPU_owned_bytes=0,load_host_s=0.,
            temporal_rephase_host_s=0.,native_cache_written=False,source_bound_once=True,
            logical_source_bytes=logical,CPU_archive_borrowed_without_copy=True,residency='one_layer_staging'))

    def dispatch(self,layer,original,q,k,v,**kwargs):
        side=self.active_side;frame=self.active_start//self.frame_tokens
        allowed=self.source_is_allowed(layer,frame)
        if not allowed:return super().dispatch(layer,original,q,k,v,**kwargs)
        cpu_pair=side['bank'][layer]
        if any(t.device.type!='cpu' for t in cpu_pair):raise RuntimeError('unexpected retained GPU source layer')
        size=sum(t.numel()*t.element_size() for t in cpu_pair)
        temporal_elements=cpu_pair[0].numel()//128*44
        # Conservative logical upper bound includes the locked rephase's FP64
        # complex input/product, BF16 changed channels and full K clone. Actual
        # allocator/process peaks are reported by the native runner separately.
        temporary=size+cpu_pair[0].numel()*cpu_pair[0].element_size()+temporal_elements*(8+8+2)+4096
        if temporary>256*1024**2:raise RuntimeError('source staging+rephase exceeds256MiB logical payload bound')
        # Separate prior GPU readiness from source copy/rephase service. These
        # explicit waits make this first capacity reference conservative.
        began=time.perf_counter();self.source_readiness_sync(q.device)
        prior=time.perf_counter()-began
        events=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
        began=time.perf_counter();events[0].record()
        with torch.inference_mode(False),torch.no_grad():
            sk,sv=self.copy_source_pair(cpu_pair,q.device)
            events[1].record()
            sk=rephase_temporal_keys(sk,side['binding']['temporal_delta'])
            events[2].record()
        events[2].synchronize();load_s=time.perf_counter()-began
        copy_ms=events[0].elapsed_time(events[1]);rephase_ms=events[1].elapsed_time(events[2])
        side['bank'][layer]=(sk,sv);side['versions'][layer]=(sk._version,sv._version)
        self.side_H2D_bytes+=size;self.side_GPU_peak_bytes=max(self.side_GPU_peak_bytes,size)
        self.side_load_host_s+=load_s;self.side_rephase_host_s+=rephase_ms/1000
        self.layer_stream_calls+=1;self.layer_stream_prior_wait_s+=prior
        self.layer_stream_copy_ms+=copy_ms;self.layer_stream_rephase_ms+=rephase_ms
        self.layer_stream_temporary_peak_bytes=max(self.layer_stream_temporary_peak_bytes,temporary)
        event=self.side_events[-1]
        event['H2D_KV_bytes']+=size;event['GPU_owned_bytes']=max(event['GPU_owned_bytes'],size)
        event['load_host_s']+=load_s;event['temporal_rephase_host_s']+=rephase_ms/1000
        try:
            output=super().dispatch(layer,original,q,k,v,**kwargs)
            self.rows[-1].update(source_layer_H2D_bytes=size,source_layer_ready_wait_host_s=prior,
                source_layer_load_host_s=load_s,source_layer_copy_stream_ms=copy_ms,
                source_layer_rephase_stream_ms=rephase_ms)
            return output
        finally:
            side['bank'][layer]=cpu_pair;side['versions'][layer]=None

    def audit(self):
        result=super().audit();a=result['immutable_source_reader']
        a.update(residency='one_layer_staging',all30_layers_onloaded_once=False,
            GPU_bank_limit_bytes=256*1024**2,source_staging_and_rephase_conservative_logical_bound_bytes=self.layer_stream_temporary_peak_bytes,
            actual_read_layer_calls=self.layer_stream_calls,source_prior_ready_wait_host_s=self.layer_stream_prior_wait_s,
            source_copy_stream_ms=self.layer_stream_copy_ms,source_rephase_stream_ms=self.layer_stream_rephase_ms,
            no_extra_CPU_raw_archive=True,no_cross_step_KV_or_output_reuse=True,
            tradeoff='lower GPU source residency in exchange for repeated H2D and temporal rephase; no speedup claim',
            timing_scope='prior ready wait separate; copy/rephase CUDA spans include host submission gaps; do not sum host and GPU spans')
        return result
