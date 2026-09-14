"""Isolate source representation while preserving the actual return read graph.

This first diagnostic retains raw archives plus committed latent snapshots.
It does not claim latent-only storage savings. Reconstruction has a separate
bounded workspace, no ordinary-generation hooks and explicit causal input
validation; it never overwrites the native window.
"""
import hashlib
import json
import time

import torch

from .immutable_source_reader import ImmutableSourceReader, request_key
from .native_commit_replay import owned_cpu, cache_metadata
from .native_retimed_memory import binding_coordinates
from .native_temporal_rephase import rephase_temporal_keys
from .history_cache import tensor_sha256


class SourceRepresentationReader(ImmutableSourceReader):
    def __init__(self,*args,source_representation,**kwargs):
        if source_representation not in ('raw_record','past_reencode','current_reencode'):
            raise ValueError('unknown source representation diagnostic')
        if (kwargs.get('source_policy')!='full_once' or kwargs.get('context_policy')!='anchor_transition'
                or kwargs.get('snapshot_window','latest8')!='latest8'
                or kwargs.get('source_stage_policy','all')!='all'):
            raise ValueError('representation fixes latest8, full_once/all, anchor transition')
        super().__init__(*args,**kwargs)
        self.representation=source_representation
        self.shared_conditioning=None
        self.pending_latent=None
        self.latent_bank={}
        self.reconstruction_active=False
        self.reconstruction_rows=[]
        self.reconstruction_events=[]
        self.representation_ledger=dict(latent_D2H_bytes=0,latent_capture_host_s=0.,
            CPU_latent_peak_bytes=0,CPU_latent_limit_bytes=64*1024**2,
            reconstruction_input_H2D_bytes=0,reconstruction_host_s=0.,
            reconstruction_prior_ready_wait_host_s=0.,reconstruction_workspace_peak_bytes=0,
            diagnostic_D2H_bytes=0)

    def _latent_storage(self):
        tensors=list(self.latent_bank.values())+([self.pending_latent] if self.pending_latent else [])
        size=sum(x['latent'].numel()*x['latent'].element_size() for x in tensors)
        self.representation_ledger['CPU_latent_peak_bytes']=max(size,self.representation_ledger['CPU_latent_peak_bytes'])
        if size>self.representation_ledger['CPU_latent_limit_bytes']:
            raise RuntimeError('committed latent diagnostic exceeds64MiB')

    def before(self,owner,values,kwargs):
        phase=float(self.pipe._dit_model.rope_temporal_offset)
        frame=int(kwargs['current_start'])//self.frame_tokens
        if self.last_phase is not None and phase!=self.last_phase:
            if self.pending_latent is None or self.pending_latent['end']!=frame:
                raise RuntimeError('phase closed without the preceding committed latent')
            self.latent_bank[frame]=self.pending_latent
            self.pending_latent=None
        super().before(owner,values,kwargs)
        ends={b['descriptor'].source_end for b in self.side_archive.banks}
        self.latent_bank={end:value for end,value in self.latent_bank.items() if end in ends}
        self._latent_storage()

    def after(self,owner,values,kwargs,result):
        clean=self.clean
        super().after(owner,values,kwargs,result)
        if clean:
            began=time.perf_counter()
            x=owned_cpu(kwargs['noisy_image_or_video'])
            end=int(kwargs['current_start'])//self.frame_tokens+x.shape[1]
            self.pending_latent=dict(end=end,latent=x,phase=self.phase)
            self.representation_ledger['latent_D2H_bytes']+=x.numel()*x.element_size()
            self.representation_ledger['latent_capture_host_s']+=time.perf_counter()-began
            self._latent_storage()

    def dispatch(self,layer,original,q,k,v,**kwargs):
        if self.reconstruction_active:
            # Called only by the explicit auxiliary forward with separate cache
            # storage. The ordinary route/owner/commit counters stay untouched.
            self.reconstruction_rows.append(dict(layer=layer,Q=q.shape[1],K=k.shape[1],
                heads=q.shape[2],logical_pairs=q.shape[1]*q.shape[2]*k.shape[1],
                backend='native_FA2',current_start=kwargs['current_start']))
            return original(q,k,v)
        return super().dispatch(layer,original,q,k,v,**kwargs)

    def _load_side(self,bank,frame,text):
        if self.representation=='raw_record':
            return super()._load_side(bank,frame,text)
        if self.shared_conditioning is None:
            raise RuntimeError('explicit shared-conditioning auxiliary guard is required')
        d=bank['descriptor'];start=d.source_end-8
        stored=self.latent_bank.get(d.source_end)
        if stored is None or stored['phase']!=d.source_phase or stored['end']>frame:
            raise RuntimeError('reconstruction source is not a matching committed past chunk')
        cpu_x=stored['latent'];device=self.pipe.kv_cache_pos[0]['k'].device
        n=8*self.frame_tokens
        resident=[sum(o is not None and o[0]=='native' and start<=o[1]<d.source_end and o[3]==d.source_phase
                      for o in owners) for owners in self.owners]
        if any(resident):raise RuntimeError('source remains in the native cache')
        native=self.pipe.kv_cache_pos
        before=[cache_metadata([c]) for c in native]
        primary_pointers=[(c['k'].data_ptr(),c['v'].data_ptr()) for c in native]
        began=time.perf_counter();torch.cuda.synchronize(device)
        self.representation_ledger['reconstruction_prior_ready_wait_host_s']+=time.perf_counter()-began
        began=time.perf_counter();aux=[];cross=[]
        with torch.inference_mode(False),torch.no_grad():
            for c in native:
                shape=(1,n,*c['k'].shape[2:])
                aux.append(dict(k=torch.zeros(shape,device=device,dtype=c['k'].dtype),
                    v=torch.zeros(shape,device=device,dtype=c['v'].dtype),
                    global_end_index=torch.tensor(start*self.frame_tokens,device=device,dtype=torch.long),
                    local_end_index=torch.tensor(0,device=device,dtype=torch.long),
                    pinned_start=torch.tensor(-1,device=device,dtype=torch.long),
                    pinned_len=torch.tensor(0,device=device,dtype=torch.long)))
                cross.append(dict(k=torch.zeros_like(self.pipe.crossattn_cache_pos[len(cross)]['k']),
                    v=torch.zeros_like(self.pipe.crossattn_cache_pos[len(cross)]['v']),is_init=False))
        owned=sum(t.numel()*t.element_size() for c in aux+cross for t in c.values() if isinstance(t,torch.Tensor))
        if owned>3*1024**3:raise RuntimeError('auxiliary cache exceeds3GiB')
        self.representation_ledger['reconstruction_workspace_peak_bytes']=max(owned,self.representation_ledger['reconstruction_workspace_peak_bytes'])
        x=cpu_x.to(device=device,copy=True)
        condition_kind='past' if self.representation=='past_reencode' else 'current'
        condition=self.shared_conditioning.auxiliary_source_condition(source_start=start,current_start=frame,kind=condition_kind)
        phase_before=self.pipe._dit_model.rope_temporal_offset
        count_before=len(self.reconstruction_rows)
        self.reconstruction_active=True
        try:
            self.pipe._dit_model.rope_temporal_offset=d.source_phase
            self.shared_conditioning.run_source_auxiliary(self.pipe,x,condition,aux,cross,
                source_start=start,current_start=frame,kind=condition_kind)
            torch.cuda.synchronize(device)
        finally:
            self.pipe._dit_model.rope_temporal_offset=phase_before
            self.reconstruction_active=False
        if len(self.reconstruction_rows)-count_before!=30:raise RuntimeError('auxiliary forward did not execute30 native layers')
        if [cache_metadata([c]) for c in native]!=before or primary_pointers!=[(c['k'].data_ptr(),c['v'].data_ptr()) for c in native]:
            raise RuntimeError('auxiliary reconstruction modified native cache metadata/storage')
        if any(int(c['global_end_index'])!=d.source_end*self.frame_tokens or int(c['local_end_index'])!=n for c in aux):
            raise RuntimeError('reconstruction did not commit exactly8 source frames')
        # Layer0 projection precedes cross/self-context mixing: independent
        # identity gate for using the exact stored clean latent/position.
        equal=[]
        for name,old in zip(('k','v'),bank['kv'][0]):
            observed=aux[0][name].cpu();self.representation_ledger['diagnostic_D2H_bytes']+=observed.numel()*observed.element_size()
            equal.append(torch.equal(observed,old))
        if not all(equal):raise RuntimeError('reconstructed layer0 differs from original source projection')
        coords=binding_coordinates(source_frame=start,target_frame=frame,frames=8,
            source_phase=d.source_phase,current_phase=self.phase,policy='recent_virtual')
        rephase_began=time.perf_counter()
        gpu=[]
        with torch.inference_mode(False),torch.no_grad():
            for c in aux:
                c['k']=rephase_temporal_keys(c['k'],coords['temporal_delta'])
                gpu.append((c['k'],c['v']))
        torch.cuda.synchronize(device);rephase_s=time.perf_counter()-rephase_began
        binding=dict(archive_version=d.archive_version,source_frames=list(range(start,d.source_end)),
                     source_phase=d.source_phase,admitted_frame=frame,**coords)
        sha=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
        bytes_=sum(t.numel()*t.element_size() for pair in gpu for t in pair)
        elapsed=time.perf_counter()-began
        source_sha=tensor_sha256(cpu_x)
        self.active_side=dict(bank=gpu,start=frame,phase=self.phase,request_key=request_key(text),binding=binding,
            binding_sha=sha,versions=[(k._version,v._version) for k,v in gpu],bytes=bytes_)
        input_bytes=x.numel()*x.element_size()
        self.side_GPU_peak_bytes=max(self.side_GPU_peak_bytes,bytes_)
        self.side_load_host_s+=elapsed;self.side_rephase_host_s+=rephase_s
        self.representation_ledger['reconstruction_input_H2D_bytes']+=input_bytes
        self.representation_ledger['reconstruction_host_s']+=elapsed
        self.side_events.append(dict(**binding,binding_sha=sha,pre_read_source_frames_per_native_layer=resident,
            source_absent_entire_native_cache=True,H2D_KV_bytes=0,GPU_owned_bytes=bytes_,load_host_s=elapsed,
            temporal_rephase_host_s=rephase_s,native_cache_written=False,source_bound_once=True,
            source_representation=self.representation,source_latent_sha256=source_sha,
            layer0_original_KV_bitwise_equal=equal,reconstruction_input_H2D_bytes=input_bytes))
        self.reconstruction_events.append(dict(source_end=d.source_end,at_frame=frame,
            condition_kind=condition_kind,source_latent_sha256=source_sha,native_cache_metadata_unchanged=True,
            native_cache_storage_unchanged=True,workspace_bytes=owned,raw_archives_still_retained=True,
            original_context_not_replayed=True,auxiliary_output_not_used_as_generated_frame=True,
            auxiliary_pairs=sum(x['logical_pairs'] for x in self.reconstruction_rows[count_before:])))

    def audit(self):
        result=super().audit()
        result['immutable_source_reader']['all30_layers_onloaded_once']=self.representation=='raw_record'
        result['immutable_source_reader']['source_bank_origin']='CPU_raw_KV' if self.representation=='raw_record' else 'computed_by_auxiliary_GPU_forward'
        result['source_representation']=dict(kind=self.representation,ledger=self.representation_ledger,
            auxiliary_attention_rows=self.reconstruction_rows,events=self.reconstruction_events,
            raw_archives_retained=True,latent_only_storage_saving_claimed=False,
            ordinary_read_pairs=sum(r['logical_pairs'] for r in self.rows),
            auxiliary_pairs=sum(r['logical_pairs'] for r in self.reconstruction_rows),
            independent_native_cache=True,auxiliary_call_uses_no_ordinary_generation_hooks=True)
        return result
