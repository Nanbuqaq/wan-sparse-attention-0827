"""Same-phase request revisions: old pin versus equal-size recent control."""
import time
import torch
from .immutable_source_reader import ImmutableSourceReader,request_key
from .return_context import contiguous_frame_runs
from .wave2_temporal_budget import updated_owners,classify_window,window_slot_indices


def revision_positions(policy,owners,physical,roles,cutoff,global_slots):
    if policy not in ('drop_pin','drop_recent'):raise ValueError('unknown revision policy')
    stale=[i for i,(slot,role) in enumerate(zip(physical,roles)) if slot>=global_slots
        and role['pin'] and not role['current'] and owners[slot][1]<cutoff]
    if not stale:return list(range(len(physical))),[]
    if policy=='drop_pin':drop=stale
    else:
        candidates=[i for i,(slot,role) in enumerate(zip(physical,roles))
            if slot>=global_slots and not role['pin'] and not role['current']]
        candidates.sort(key=lambda i:owners[physical[i]][1])
        if len(candidates)<len(stale):raise RuntimeError('equal recent-removal budget unavailable; do not remove current/global')
        drop=candidates[:len(stale)]
    return [i for i in range(len(physical)) if i not in drop],drop


class RequestPinRead(ImmutableSourceReader):
    def __init__(self,*args,request_pin_policy,**kwargs):
        super().__init__(*args,**kwargs)
        if (request_pin_policy not in ('drop_pin','drop_recent') or self.source_policy!='off'
            or self.source_archive_enabled or self.context_policy!='full'):
            raise ValueError('revision diagnostic requires full native context, fixed-off and no archive')
        self.request_pin_policy=request_pin_policy
        self.request_frame=None;self.request_phase=None;self.request_digest=None
        self.revision_start=None;self.revision_events=[];self.revision_times=[]

    def before(self,owner,values,kwargs):
        super().before(owner,values,kwargs)
        frame=self.active_start//self.frame_tokens
        if frame==self.request_frame:return
        digest=request_key(self.current_text(frame))
        if self.phase!=self.request_phase:self.revision_start=None
        elif digest!=self.request_digest:
            self.revision_start=frame
            self.revision_events.append(dict(frame=frame,phase=self.phase,current_request_sha256=digest,
                previous_request_sha256=self.request_digest,semantic_operation_inferred=False))
            if len(self.revision_events)>2048:raise RuntimeError('bounded request-event record exceeded')
        self.request_frame,self.request_phase,self.request_digest=frame,self.phase,digest

    def dispatch(self,layer,original,q,k,v,**kwargs):
        if self.revision_start is None:return super().dispatch(layer,original,q,k,v,**kwargs)
        began=time.perf_counter();info=kwargs['info']
        owners=updated_owners(self.owners[layer],info,kwargs['current_start'],self.frame_tokens,self.calls,self.phase)
        physical=window_slot_indices(end=kwargs['cache_end'],start=kwargs['window_start'],effective_sink=kwargs['effective_sink'],
            pinned_start=kwargs['pinned_start'],pinned_len=kwargs['pinned_len'],prepend_sink=kwargs['prepend_sink'],
            prepend_pinned=kwargs['prepend_pinned'],max_tokens=kwargs['max_tokens'],frame_tokens=self.frame_tokens)
        roles=classify_window(owners,physical,info,self.frame_tokens,kwargs['effective_sink'],kwargs['global_sink_tokens'],kwargs['pinned_start'],kwargs['pinned_len'])
        keep,drop=revision_positions(self.request_pin_policy,owners,physical,roles,self.revision_start,kwargs['global_sink_tokens']//self.frame_tokens)
        if not drop:return super().dispatch(layer,original,q,k,v,**kwargs)
        runs=contiguous_frame_runs(keep);mask_s=time.perf_counter()-began
        events=[torch.cuda.Event(enable_timing=True) for _ in range(2)];pack_s=0.
        def execute(qq,kk,vv):
            nonlocal pack_s
            events[0].record();start=time.perf_counter()
            pk=torch.cat([kk[:,a*self.frame_tokens:b*self.frame_tokens] for a,b in runs],dim=1)
            pv=torch.cat([vv[:,a*self.frame_tokens:b*self.frame_tokens] for a,b in runs],dim=1)
            pack_s=time.perf_counter()-start;events[1].record()
            return original(qq,pk,pv)
        output=super().dispatch(layer,execute,q,k,v,**kwargs)
        row=self.rows[-1];n=len(keep)*self.frame_tokens
        row['native_protected_union_tokens']=row['protected_union_tokens']
        row['native_pin_tokens']=row['pin_tokens']
        row['native_optional_tokens']=row['optional_tokens']
        kept_roles=[roles[i] for i in keep]
        protected=sum(r['current'] or r['sink'] or r['pin'] for r in kept_roles)*self.frame_tokens
        row.update(request_pin_policy=self.request_pin_policy,request_revision_start=self.revision_start,
            request_excluded_native_frames=[owners[physical[i]][1] for i in drop],
            actual_K=n,logical_pairs=q.shape[1]*q.shape[2]*n,non_source_pairs=q.shape[1]*q.shape[2]*n,
            GPU_gather_output_bytes=2*n*k.shape[2]*k.shape[3]*k.element_size(),
            request_filter_host_s=mask_s,request_pack_host_s=pack_s,backend='native_FA2_request_revision_filter',
            pin_tokens=sum(r['pin'] for r in kept_roles)*self.frame_tokens,
            sink_tokens=sum(r['sink'] for r in kept_roles)*self.frame_tokens,
            protected_union_tokens=protected,optional_tokens=n-protected,selected_optional_tokens=n-protected)
        row['prepare_host_s']+=mask_s+pack_s
        self.revision_times.append((row,events))
        if len(self.revision_times)>2048:raise RuntimeError('bounded request timing records exceeded')
        return output

    def audit(self):
        result=super().audit()
        if self.revision_times:
            torch.cuda.synchronize()
            for row,ev in self.revision_times:row['request_pack_stream_ms']=ev[0].elapsed_time(ev[1])
        result['request_pin_read']=dict(policy=self.request_pin_policy,events=self.revision_events,
            filtered_layer_calls=len(self.revision_times),source_archive=False,
            only_current_arrived_text_and_native_owners=True,current_and_global_always_kept=True,
            cache_writes_and_RoPE_unchanged=True,reset_on_actual_scene_phase_change=True,
            no_semantic_request_classifier=True)
        return result
