"""Request-conditioned differences anchored in original causal source KV.

This is an experimental intervention, not an exact causal decomposition.
Both directions pay for two isolated reconstructions. Raw CPU banks remain.
"""
import time
import torch
from .source_representation_reader import SourceRepresentationReader
from .native_temporal_rephase import rephase_temporal_keys


def apply_condition_delta(raw,current,past,sign):
    if sign not in (-1,1) or raw.shape!=current.shape or raw.shape!=past.shape:
        raise ValueError('matched source tensors and a fixed unit direction required')
    if any(x.dtype!=torch.bfloat16 or x.device!=raw.device for x in (raw,current,past)):
        raise ValueError('matched BF16 source representation required')
    # Form the difference FIRST: identical reconstructions give exactly zero,
    # without the rounding error of (raw + current) - past.
    return (raw.float()+sign*(current.float()-past.float())).to(raw.dtype)


class ConditionalSourceDelta(SourceRepresentationReader):
    def __init__(self,*args,delta_direction,**kwargs):
        if delta_direction not in ('forward','reverse','null'):
            raise ValueError('unregistered conditional difference direction')
        if kwargs.get('source_representation')!='raw_record':
            raise ValueError('conditional delta preserves the original raw anchor')
        super().__init__(*args,**kwargs)
        self.delta_direction=delta_direction
        self.delta_ledger=dict(direction=delta_direction,raw_H2D_bytes=0,correction_host_s=0.,
            source_workspace_limit_bytes=6*1024**3,source_workspace_conservative_bound_bytes=0,
            source_workspace_bound_excludes_ordinary_DiT_activations=True,corrected_layers=0)
        self.delta_timings=[]

    def _load_side(self,bank,frame,text):
        size=sum(t.numel()*t.element_size() for pair in bank['kv'] for t in pair)
        bound=2*size+512*1024**2
        if bound>self.delta_ledger['source_workspace_limit_bytes']:
            raise RuntimeError('two reconstructions plus one-layer correction exceed6GiB source workspace')
        self.delta_ledger['source_workspace_conservative_bound_bytes']=max(bound,self.delta_ledger['source_workspace_conservative_bound_bytes'])
        first_event=len(self.side_events)
        original_kind=self.representation
        try:
            self.representation='past_reencode'
            super()._load_side(bank,frame,text)
            past=self.active_side
            self.representation='past_reencode' if self.delta_direction=='null' else 'current_reencode'
            super()._load_side(bank,frame,text)
            current=self.active_side
        finally:
            self.representation=original_kind
        if past['binding_sha']!=current['binding_sha']:
            raise RuntimeError('conditional correction source bindings differ')
        events=self.side_events[first_event:]
        if len(events)!=2:raise RuntimeError('expected two explicitly charged reconstructions')
        device=current['bank'][0][0].device
        torch.cuda.synchronize(device);started=time.perf_counter()
        corrected=[];sign=-1 if self.delta_direction=='reverse' else 1
        with torch.inference_mode(False),torch.no_grad():
            for layer,(cpu_k,cpu_v) in enumerate(bank['kv']):
                timing=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
                timing[0].record()
                rawk=cpu_k.to(device=device,copy=True);rawv=cpu_v.to(device=device,copy=True)
                timing[1].record()
                rawk=rephase_temporal_keys(rawk,current['binding']['temporal_delta'])
                timing[2].record()
                kp,vp=past['bank'][layer];kc,vc=current['bank'][layer]
                k=apply_condition_delta(rawk,kc,kp,sign);v=apply_condition_delta(rawv,vc,vp,sign)
                timing[3].record();self.delta_timings.append((layer,timing))
                corrected.append((k,v))
                past['bank'][layer]=None;current['bank'][layer]=None
                self.delta_ledger['corrected_layers']+=1
        torch.cuda.synchronize(device);elapsed=time.perf_counter()-started
        self.delta_ledger['raw_H2D_bytes']+=size;self.delta_ledger['correction_host_s']+=elapsed
        self.side_H2D_bytes+=size;self.side_load_host_s+=elapsed
        self.active_side=dict(current,bank=corrected,versions=[(k._version,v._version) for k,v in corrected])
        del self.side_events[first_event:]
        event=dict(events[-1],source_representation='raw_plus_conditional_difference',
            delta_direction=self.delta_direction,H2D_KV_bytes=size,
            reconstruction_input_H2D_bytes=sum(e['reconstruction_input_H2D_bytes'] for e in events),
            load_host_s=sum(e['load_host_s'] for e in events)+elapsed,
            temporal_rephase_host_s=sum(e['temporal_rephase_host_s'] for e in events),
            auxiliary_forward_count=2,raw_context_anchor_preserved=True,
            zero_delta_expected_to_reproduce_raw=self.delta_direction=='null')
        self.side_events.append(event)

    def audit(self):
        result=super().audit()
        if self.delta_timings:torch.cuda.synchronize()
        rows=[dict(layer=layer,raw_copy_stream_ms=e[0].elapsed_time(e[1]),
                   raw_rephase_stream_ms=e[1].elapsed_time(e[2]),delta_stream_ms=e[2].elapsed_time(e[3]))
              for layer,e in self.delta_timings]
        result['conditional_source_delta']=dict(**self.delta_ledger,timings=rows,
            formula='BF16(FP32(raw) + sign * (FP32(current) - FP32(past))) in the same bound coordinates',
            sign=-1 if self.delta_direction=='reverse' else 1,
            second_condition='past' if self.delta_direction=='null' else 'current',
            delta_normalized_or_clipped=False,no_future_outputs_or_observed_state_labels=True,
            timing_scope='CUDA stream spans include host submission gaps; not additive with host spans',
            actual_process_GPU_peak_reported_by_runner=True)
        result['source_representation']['kind']='raw_plus_conditional_difference'
        result['immutable_source_reader']['all30_layers_onloaded_once']=True
        result['immutable_source_reader']['source_bank_origin']='original raw onload plus two explicitly charged auxiliary reconstructions'
        return result
