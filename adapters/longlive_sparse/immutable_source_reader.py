"""Bounded source side-bank: original native cache is never overwritten.

The primary reference explicitly concatenates the permitted native and source
KV. The two-partial LSE backend remains an explicit diagnostic. Source timing
and layer budgets include clean. The bank is bound once per admitted request;
later chunks keep the same coordinates rather than refreshing its apparent age.
"""
import hashlib
import json
import time

import torch

from .access_motion_memory import BoundedSceneArchive
from .native_retimed_memory import binding_coordinates
from .native_scene_admission import choose_scene, has_revisit_cue
from .native_temporal_rephase import rephase_temporal_keys
from .source_layer_budget import LAYERRECALL_PRIOR
from .wave2_temporal_budget import Wave2TemporalBudget

POLICIES=('off','full_once','prior_once','full_three','prior_three')


def source_allowed(policy, layer, age_chunks):
    if policy not in POLICIES or not 0<=layer<30 or age_chunks<0:
        raise ValueError('invalid source lifetime policy, layer or causal age')
    if policy=='off':return False
    lifetime=3 if policy.endswith('three') else 1
    return age_chunks<lifetime and (policy.startswith('full') or layer in LAYERRECALL_PRIOR)


def request_key(text):
    return hashlib.sha256(text.removeprefix('The scene transitions. ').strip().encode()).hexdigest()


def side_attention(q, k, v, sk, sv):
    from flash_attn import flash_attn_func
    if sk.shape!=sv.shape or k.shape!=v.shape or sk.dtype!=q.dtype or sk.shape[2:]!=q.shape[2:]:
        raise ValueError('side-bank shape or dtype differs')
    outputs=[];lses=[]
    for kk,vv in ((k,v),(sk,sv)):
        out,lse,prob=flash_attn_func(q,kk,vv,dropout_p=0.,causal=False,return_attn_probs=True)
        if prob.numel():raise RuntimeError('unexpected materialized attention probability matrix')
        outputs.append(out);lses.append(lse)
    mass=(lses[1]-lses[0]).sigmoid()
    weight=mass.transpose(1,2).unsqueeze(-1)
    out=((1-weight)*outputs[0].float()+weight*outputs[1].float()).to(q.dtype)
    return out,mass


class SideArchive(BoundedSceneArchive):
    def before(self,owner,values,kwargs,*,current_text,allow_select=True):
        """Produce/select only; deliberately never call native _install."""
        frame=int(kwargs['current_start'])//self.pipe.frame_seq_length
        count=self.counts.get(frame,0);self.counts[frame]=count+1
        if count:return None
        for old in sorted(self.counts)[:-2]:del self.counts[old]
        if max(len(self.archives),len(self.decisions))>=2048:raise RuntimeError('finite source event bound reached')
        phase=float(self.pipe._dit_model.rope_temporal_offset)
        if self.last_commit is not None and self.last_commit['phase']!=phase:self._archive_last_scene(frame)
        if not allow_select:return None
        prototype=self._prototype(kwargs['conditional_dict']) if has_revisit_cue(current_text) else None
        began=time.perf_counter()
        decision=choose_scene(current_text,prototype,[b['descriptor'] for b in self.banks],frame,**self.policy)
        self.ledger['selector_CPU_wall_s']+=time.perf_counter()-began
        decision.update(at_latent=frame,current_request_key=request_key(current_text))
        self.decisions.append(decision)
        return next((b for b in self.banks if b['descriptor'].archive_version==decision['selected_version']),None)


class ImmutableSourceReader(Wave2TemporalBudget):
    def __init__(self,pipe,method,*,source_policy,source_replay=False,source_backend='concat',context_policy='full',source_order='append',snapshot_window='latest8',**kwargs):
        if method!='w2_full_recall' or kwargs.get('version_policy') is not None:
            raise ValueError('source lifetime is isolated from version and steady routing')
        if source_policy not in POLICIES:raise ValueError('unknown source policy')
        if source_backend not in ('concat','partial'):raise ValueError('unknown explicit source backend')
        from .return_context import POLICIES as CONTEXT_POLICIES
        if context_policy not in CONTEXT_POLICIES:raise ValueError('unknown native context control')
        if context_policy!='full' and source_backend!='concat':raise ValueError('context controls use one explicit concat backend')
        if source_order not in ('append','after_global'):raise ValueError('unknown explicit source packing order')
        if pipe._dit_model.t_scale!=1 or pipe._dit_model.rope_method!='linear' or pipe._dit_model.original_seq_len is not None:
            raise ValueError('source lifetime requires the qualified absolute native RoPE')
        super().__init__(pipe,method,**kwargs)
        self.scene=None  # No native-slot installation and no second set of hooks.
        self.side_archive=SideArchive(pipe,archive_budget=8*1024**3)
        if snapshot_window not in ('latest8','oldest_resident8','request_resident8'):raise ValueError('unknown source snapshot window')
        if snapshot_window!='latest8':
            from .resident_snapshot import ResidentSnapshotArchive
            self.side_archive=ResidentSnapshotArchive(pipe,archive_budget=8*1024**3,
                owner_provider=lambda:self.owners,window=snapshot_window)
        self.snapshot_window=snapshot_window
        self.source_policy=source_policy;self.source_replay=source_replay
        self.source_backend=source_backend;self.source_timings=[]
        self.context_policy=context_policy;self.return_start=None;self.return_phase=None
        self.source_order=source_order
        self.active_side=None;self.served_phases=set();self.side_events=[];self.side_expirations=[]
        self.side_H2D_bytes=0;self.side_GPU_peak_bytes=0;self.side_load_host_s=0.;self.side_rephase_host_s=0.
        self.side_numeric=[];self.sample_mass=[]

    def _load_side(self,bank,frame,text):
        descriptor=bank['descriptor'];source_start=descriptor.source_end-8
        coordinates=binding_coordinates(source_frame=source_start,target_frame=frame,frames=8,
            source_phase=descriptor.source_phase,current_phase=self.phase,policy='recent_virtual')
        residency=[sum(o is not None and o[0]=='native' and source_start<=o[1]<descriptor.source_end
                       and o[3]==descriptor.source_phase for o in owners) for owners in self.owners]
        if any(residency):raise RuntimeError('pilot source has not left the entire native cache')
        gpu=[];bytes_=0
        torch.cuda.synchronize();began=time.perf_counter();rephase_s=0.
        for layer,(cpu_k,cpu_v) in enumerate(bank['kv']):
            device=self.pipe.kv_cache_pos[layer]['k'].device
            # Native inference tensors have no mutation counter. Our owned
            # copies deliberately retain one; this does not enable gradients.
            with torch.inference_mode(False),torch.no_grad():
                sk=cpu_k.to(device=device,copy=True);sv=cpu_v.to(device=device,copy=True)
                torch.cuda.synchronize();start=time.perf_counter()
                sk=rephase_temporal_keys(sk,coordinates['temporal_delta'])
            torch.cuda.synchronize();rephase_s+=time.perf_counter()-start
            gpu.append((sk,sv));bytes_+=cpu_k.numel()*cpu_k.element_size()+cpu_v.numel()*cpu_v.element_size()
        torch.cuda.synchronize();elapsed=time.perf_counter()-began
        if bytes_>3*1024**3:raise RuntimeError('side GPU bank exceeds registered3GiB limit')
        binding=dict(archive_version=descriptor.archive_version,source_frames=list(range(source_start,descriptor.source_end)),
                     source_phase=descriptor.source_phase,admitted_frame=frame,**coordinates)
        binding_sha=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
        self.active_side=dict(bank=gpu,start=frame,phase=self.phase,request_key=request_key(text),binding=binding,
            binding_sha=binding_sha,versions=[(k._version,v._version) for k,v in gpu],bytes=bytes_)
        self.side_events.append(dict(**binding,binding_sha=binding_sha,pre_read_source_frames_per_native_layer=residency,
            source_absent_entire_native_cache=True,H2D_KV_bytes=bytes_,GPU_owned_bytes=bytes_,load_host_s=elapsed,
            temporal_rephase_host_s=rephase_s,native_cache_written=False,source_bound_once=True))
        self.side_H2D_bytes+=bytes_;self.side_load_host_s+=elapsed;self.side_rephase_host_s+=rephase_s
        self.side_GPU_peak_bytes=max(self.side_GPU_peak_bytes,bytes_)

    def before(self,owner,values,kwargs):
        previous=self.last_phase
        super().before(owner,values,kwargs)
        frame=self.active_start//self.frame_tokens;text=self.current_text(frame)
        if previous!=self.phase:
            self.return_start=frame if has_revisit_cue(text) else None
            self.return_phase=self.phase if self.return_start is not None else None
        if self.active_side is not None:
            side=self.active_side;age=(frame-side['start'])//8
            lifetime=3 if self.source_policy.endswith('three') else 1
            reason=('phase_changed' if side['phase']!=self.phase else 'current_request_changed'
                    if side['request_key']!=request_key(text) else 'lease_expired' if age>=lifetime else None)
            if reason:
                self.side_expirations.append(dict(frame=frame,reason=reason,binding_sha=side['binding_sha']))
                self.active_side=None
        bank=self.side_archive.before(owner,values,kwargs,current_text=text,allow_select=self.phase not in self.served_phases)
        if bank is not None and self.phase not in self.served_phases:
            self.served_phases.add(self.phase)
            if len(self.served_phases)>2048:raise RuntimeError('finite phase metadata bound reached')
            if self.source_policy!='off':self._load_side(bank,frame,text)

    def dispatch(self,layer,original,q,k,v,**kwargs):
        side=self.active_side;frame=self.active_start//self.frame_tokens
        allowed=side is not None and source_allowed(self.source_policy,layer,(frame-side['start'])//8)
        context_started=time.perf_counter();positions=list(range(k.shape[1]//self.frame_tokens));excluded=[]
        if self.context_policy!='full' and self.return_phase==self.phase:
            from .wave2_temporal_budget import updated_owners,classify_window,window_slot_indices
            from .return_context import permitted_native_frames
            info=kwargs['info']
            owners=updated_owners(self.owners[layer],info,kwargs['current_start'],self.frame_tokens,self.calls,self.phase)
            physical=window_slot_indices(end=kwargs['cache_end'],start=kwargs['window_start'],effective_sink=kwargs['effective_sink'],
                pinned_start=kwargs['pinned_start'],pinned_len=kwargs['pinned_len'],prepend_sink=kwargs['prepend_sink'],
                prepend_pinned=kwargs['prepend_pinned'],max_tokens=kwargs['max_tokens'],frame_tokens=self.frame_tokens)
            roles=classify_window(owners,physical,info,self.frame_tokens,kwargs['effective_sink'],kwargs['global_sink_tokens'],kwargs['pinned_start'],kwargs['pinned_len'])
            if len(physical)*self.frame_tokens!=k.shape[1]:raise RuntimeError('context role/window geometry differs')
            positions=permitted_native_frames(self.context_policy,owners,physical,roles,self.phase,
                first_return=frame==self.return_start,returning=True,global_slots=kwargs['global_sink_tokens']//self.frame_tokens)
            excluded=[owners[slot][1] for i,slot in enumerate(physical) if i not in positions]
        native_tokens=len(positions)*self.frame_tokens;context_changed=native_tokens!=k.shape[1]
        from .return_context import contiguous_frame_runs
        runs=contiguous_frame_runs(positions);context_prepare_s=time.perf_counter()-context_started
        mass=None;sk=sv=None;events=None;pack_host_s=attention_host_s=0.
        def execute(qq,kk,vv):
            nonlocal mass,sk,sv,events,pack_host_s,attention_host_s
            if not allowed and not context_changed:return original(qq,kk,vv)
            if allowed:
                sk,sv=side['bank'][layer]
                if (sk._version,sv._version)!=side['versions'][layer]:raise RuntimeError('immutable source mutated')
            events=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
            events[0].record();began=time.perf_counter()
            if self.source_backend=='concat':
                parts_k=[kk[:,a*self.frame_tokens:b*self.frame_tokens] for a,b in runs]
                parts_v=[vv[:,a*self.frame_tokens:b*self.frame_tokens] for a,b in runs]
                if allowed and self.source_order=='after_global':
                    g=kwargs['global_sink_tokens']//self.frame_tokens
                    if positions[:g]!=list(range(g)):raise RuntimeError('source order needs the intact native global prefix')
                    rest=contiguous_frame_runs(positions[g:])
                    packed_k=torch.cat([kk[:,:g*self.frame_tokens],sk]+[kk[:,a*self.frame_tokens:b*self.frame_tokens] for a,b in rest],dim=1)
                    packed_v=torch.cat([vv[:,:g*self.frame_tokens],sv]+[vv[:,a*self.frame_tokens:b*self.frame_tokens] for a,b in rest],dim=1)
                else:
                    packed_k=torch.cat(parts_k+([sk] if allowed else []),dim=1)
                    packed_v=torch.cat(parts_v+([sv] if allowed else []),dim=1)
                pack_host_s=time.perf_counter()-began;events[1].record();began=time.perf_counter()
                out=original(qq,packed_k,packed_v)
            else:
                events[1].record();began=time.perf_counter()
                out,mass=side_attention(qq,kk,vv,sk,sv)
            attention_host_s=time.perf_counter()-began;events[2].record()
            if self.source_replay and allowed and not self.side_numeric:
                from flash_attn import flash_attn_func
                fullk=packed_k if self.source_backend=='concat' else torch.cat([kk,sk],dim=1)
                fullv=packed_v if self.source_backend=='concat' else torch.cat([vv,sv],dim=1)
                reference=flash_attn_func(qq,fullk,fullv,dropout_p=0.,causal=False)
                error=(out.float()-reference.float()).square().sum()/reference.float().square().sum().clamp_min(1e-30)
                rel=float(error.sqrt());maximum=float((out.float()-reference.float()).abs().max())
                self.side_numeric.append(dict(frame=frame,layer=layer,relative_L2=rel,max_abs=maximum,
                    actual_q_tokens=q.shape[1],native_k_tokens=native_tokens,source_tokens=sk.shape[1],
                    teacher_used_for_routing=False,full_Q_reference=True,backend=self.source_backend))
                if self.source_backend=='concat':
                    native_k=torch.cat(parts_k,dim=1) if context_changed else kk
                    native_v=torch.cat(parts_v,dim=1) if context_changed else vv
                    partial,_=side_attention(qq,native_k,native_v,sk,sv)
                    partial_rel=float(((partial.float()-reference.float()).square().sum()/reference.float().square().sum().clamp_min(1e-30)).sqrt())
                    partial_max=float((partial.float()-reference.float()).abs().max())
                    self.side_numeric[-1]['isolated_partial_diagnostic']=dict(relative_L2=partial_rel,max_abs=partial_max,
                        passed_original_threshold=partial_rel<=.01 and partial_max<=.02,affected_generation=False)
                if rel>.01 or maximum>.02:raise RuntimeError(f'side backend numerical gate failed: relative={rel}, max_abs={maximum}')
            return out
        output=super().dispatch(layer,execute,q,k,v,**kwargs)
        row=self.rows[-1];source_tokens=sk.shape[1] if allowed else 0
        native_pairs=q.shape[1]*q.shape[2]*native_tokens
        row.update(source_policy=self.source_policy,source_side_reader=True,source_age_chunks=None if side is None else (frame-side['start'])//8,
            admitted_source_visible_tokens=source_tokens,source_native_cache_written=False,
            side_binding_sha=None if side is None else side['binding_sha'],
            non_source_pairs=native_pairs,logical_pairs=native_pairs+q.shape[1]*q.shape[2]*source_tokens,
            actual_K=native_tokens+source_tokens,source_pairs=q.shape[1]*q.shape[2]*source_tokens,
            context_policy=self.context_policy,context_excluded_native_frames=excluded,
            context_filter_prepare_host_s=context_prepare_s,
            source_pack_host_s=pack_host_s,source_attention_submit_host_s=attention_host_s,
            GPU_gather_output_bytes=2*(native_tokens+source_tokens)*k.shape[2]*k.shape[3]*k.element_size()
                if (allowed or context_changed) and self.source_backend=='concat' else 0,
            backend=('native_FA2_explicit_side_concat' if self.source_backend=='concat' else 'native_FA2_two_bank_LSE_merge') if allowed else 'native_FA2_context_concat' if context_changed else 'native_FA2')
        row['prepare_host_s']+=context_prepare_s
        if events is not None:
            self.source_timings.append((row,events))
            if len(self.source_timings)>2048:raise RuntimeError('bounded source timing capacity2048 reached')
        if mass is not None and layer==4:self.sample_mass.append((row,mass.mean().detach()))
        return output

    def after(self,owner,values,kwargs,result):
        super().after(owner,values,kwargs,result)
        self.side_archive.after(owner,values,kwargs,result)

    def audit(self):
        result=super().audit()
        if self.source_timings:
            torch.cuda.synchronize()
            for row,events in self.source_timings:
                row.update(source_pack_stream_ms=events[0].elapsed_time(events[1]),
                           source_attention_stream_ms=events[1].elapsed_time(events[2]))
        if self.source_replay and not self.side_numeric:raise RuntimeError('registered source replay was not reached')
        if self.sample_mass:
            values=torch.stack([x[1] for x in self.sample_mass]).cpu().tolist()
            for (row,_),value in zip(self.sample_mass,values):row['sampled_layer4_source_mass_mean']=value
        result.update(scene=self.side_archive.audit(),immutable_source_reader=dict(policy=self.source_policy,backend=self.source_backend,context_policy=self.context_policy,source_order=self.source_order,snapshot_window=self.snapshot_window,
            admissions=self.side_events,expirations=self.side_expirations,H2D_KV_bytes=self.side_H2D_bytes,
            GPU_bank_peak_bytes=self.side_GPU_peak_bytes,GPU_bank_limit_bytes=3*1024**3,
            CPU_archive_limit_bytes=8*1024**3,source_load_host_s=self.side_load_host_s,
            temporal_rephase_host_s=self.side_rephase_host_s,numerical_replay=self.side_numeric,
            mass_audit_D2H_bytes=len(self.sample_mass)*4,native_slot_installation=False,
            clean_obeys_same_source_policy=True,all30_layers_onloaded_once=bool(self.side_events),
            timing_scope='CUDA stream spans include CPU submission gaps; do not add them to host spans',
            same_graph_concat_equivalence_is_numerical_not_bitwise=self.source_backend=='partial'),
            source_hashes=[dict(archive_version=a['archive_version'],source_end=a['source_end'],
                own_clean_latent_sha256=self.clean_latent_hashes.get(a['source_end'])) for a in self.side_archive.archives],
            one_attention_dispatch_per_layer=True,FA2_calls_when_source_allowed=2 if self.source_backend=='partial' else 1)
        return result
