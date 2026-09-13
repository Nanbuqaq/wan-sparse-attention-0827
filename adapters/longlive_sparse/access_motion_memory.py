"""One coordinator for explicit eligibility x bounded external restoration.

The protected scene controller is reused without attaching separate hooks.
No diagnostic retired payload is allocated. Native rolling/pin and the existing
source rephase stay unchanged. This is a controlled baseline, not entity tracking.
"""
from .native_scene_release import NativeSceneRelease
from .native_causal_scene_memory import NativeCausalSceneMemory
from .native_scene_admission import has_revisit_cue


def decode_owner(owner):
    if owner is None:raise ValueError('missing physical owner')
    if len(owner)==4 and owner[0]=='native':
        return dict(kind='native',frame=owner[1],epoch=owner[2],phase=owner[3],binding=None)
    if len(owner)==7 and owner[0]=='recalled':
        return dict(kind='recalled',frame=owner[1],archive_version=owner[2],binding=owner[3],
                    source_phase=owner[4],virtual_frame=owner[5],phase=owner[6])
    raise ValueError('unknown typed owner layout')


def eligible_positions(owners,physical,phase,active,admitted):
    selected=[]
    for position,slot in enumerate(physical):
        owner=owners[slot];d=decode_owner(owner)
        if not active or (d['kind']=='native' and d['phase']==phase) or owner in admitted:
            selected.append(position)
    return selected


def return_context_positions(narrow,physical,scope,global_slots):
    if scope=='anchor':return sorted(set(narrow)|{i for i,slot in enumerate(physical) if slot<global_slots})
    if scope=='no_global':return [i for i,slot in enumerate(physical) if slot>=global_slots]
    return narrow


class BoundedSceneArchive(NativeCausalSceneMemory):
    """Cap diagnostic metadata as well as the existing byte-bounded raw bank."""
    def before(self,*args,**kwargs):
        if max(len(self.archives),len(self.decisions),len(self.installations))>=2048:
            raise RuntimeError('registered scene metadata capacity2048 reached')
        super().before(*args,**kwargs)
        if len(self.counts)>2:
            for frame in sorted(self.counts)[:-2]:del self.counts[frame]

    def _archive_last_scene(self,current_frame):
        super()._archive_last_scene(current_frame)
        bank=self.banks[-1];storages={};logical=0
        for pair in bank['kv']:
            for tensor in pair:
                storage=tensor.untyped_storage();storages[storage.data_ptr()]=storage.nbytes()
                logical+=tensor.numel()*tensor.element_size()
        owned=sum(storages.values())
        if owned!=logical:raise RuntimeError('raw snapshot must own compact storage, not resident-window view')
        self.archives[-1].update(unique_owned_raw_storage_bytes=owned,logical_tensor_bytes=logical,
            storage_count=len(storages),ready=True)


class AccessMotionMemory(NativeSceneRelease):
    def __init__(self,pipe,method,*,restore=False,return_scope='broad',source_beta=None,weight_replay=False,**kwargs):
        if return_scope not in ('broad','narrow','anchor','no_global'):raise ValueError('unknown return eligibility')
        super().__init__(pipe,method,retired_copy=False,**kwargs)
        self.scene=BoundedSceneArchive(pipe) if restore else None
        self.restore_enabled=restore;self.return_scope=return_scope;self.admitted=set()
        self.source_residency=[];self.returning=False
        if source_beta is not None and (not restore or source_beta not in (.5,1.,2.)):
            raise ValueError('source beta requires explicit restored source')
        self.source_beta=source_beta;self.weight_replay=weight_replay;self.weight_partitions={};self.weight_diagnostics=[]
        self.weight_index_H2D_bytes=0;self.weight_prepare_host_s=0.
        self.global_slot_count=0

    def before(self,owner,values,kwargs):
        # Capture ownership before any archive installation, including physical
        # cache slots outside the last visible window. Zero here is stronger
        # than zero in the visible subset, and cannot be inferred from catalog.
        prior_owners=[list(x) for x in self.owners]
        prior=len(self.storage_events);previous=self.last_phase
        super().before(owner,values,kwargs)
        if previous!=self.phase:
            self.admitted.clear()
            self.returning=has_revisit_cue(self.current_text(self.active_start//self.frame_tokens))
        for event in self.storage_events[prior:]:
            first,last=event['slot_range']
            for layer in self.owners:self.admitted.update(layer[first:last])
            frames=set(event['source_frames']);source_phase=event['source_phase']
            resident=[sum(o is not None and decode_owner(o)['frame'] in frames and
                (decode_owner(o).get('source_phase',decode_owner(o)['phase'])==source_phase)
                for o in layer) for layer in prior_owners]
            self.source_residency.append(dict(frame=event['frame'],source_frames=event['source_frames'],
                archive_version=event['archive_version'],binding=event['storage_version'],
                pre_install_all_cache_source_frames_per_layer=resident,
                source_absent_entire_native_cache=all(x==0 for x in resident)))
        # Initial phase retains original native graph; every subsequent phase
        # filters even clean calls unless a broad return explicitly permits it.
        if self.release_events:
            self.release_phase=self.phase
            self.release_active=not (self.returning and self.return_scope=='broad')
            event=self.release_events[-1]
            if event['frame']==self.active_start//self.frame_tokens:
                event.update(archive_restored=len(self.storage_events)>prior,release_active=self.release_active,
                    return_scope=self.return_scope,archive_kind='bounded_clean8' if self.restore_enabled else 'none')

    def allowed_positions(self,owners,physical):
        admitted=self.admitted if self.returning else set()
        selected=eligible_positions(owners,physical,self.release_phase,self.release_active,admitted)
        if self.returning and self.release_events:
            selected=return_context_positions(selected,physical,self.return_scope,self.global_slot_count)
        self.admitted_visible_tokens=sum(owners[physical[i]] in admitted for i in selected)*self.frame_tokens
        self.source_permitted_frame_positions=tuple(j for j,i in enumerate(selected) if owners[physical[i]] in admitted)
        if len(self.source_residency)>2048:raise RuntimeError('source audit capacity2048 reached')
        return selected

    def dispatch(self,layer,original,q,k,v,**kwargs):
        self.global_slot_count=kwargs['global_sink_tokens']//self.frame_tokens
        from .source_weight import weighted_source_attention,independent_sample_error
        import torch,time
        applied=False
        def weighted(qq,kk,vv):
            nonlocal applied
            frames=self.source_permitted_frame_positions
            if self.source_beta is None or not frames:return original(qq,kk,vv)
            began=time.perf_counter();key=(frames,kk.shape[1],str(kk.device));partition=self.weight_partitions.get(key)
            if partition is None:
                ft=self.frame_tokens;chosen=set(frames);source=[t for f in frames for t in range(f*ft,(f+1)*ft)]
                other=[t for f in range(kk.shape[1]//ft) if f not in chosen for t in range(f*ft,(f+1)*ft)]
                partition=tuple(torch.tensor(x,device=kk.device,dtype=torch.long) for x in (source,other))
                self.weight_partitions[key]=partition;self.weight_index_H2D_bytes+=sum(x.numel()*x.element_size() for x in partition)
                if len(self.weight_partitions)>32:raise RuntimeError('source weight geometry bound32 exceeded')
            self.weight_prepare_host_s+=time.perf_counter()-began
            out,mass=weighted_source_attention(qq,kk,vv,*partition,self.source_beta);applied=True
            if layer==0 and self.weight_replay and not self.weight_diagnostics:
                from .history_cache import tensor_sha256
                error=independent_sample_error(qq,kk,vv,partition[0],self.source_beta,out)
                if error>.02:raise RuntimeError('independent weighted-source FP32 gate failed')
                self.weight_diagnostics.append(dict(frame=self.active_start//self.frame_tokens,layer=layer,
                    source_tokens=len(partition[0]),permitted_tokens=kk.shape[1],beta=self.source_beta,
                    source_mass_mean=float(mass.mean()),FP32_sample_relative_L2=error,
                    teacher_is_isolated=True,sites=32,input_QKV_sha256=[tensor_sha256(x) for x in (qq,kk,vv)],
                    diagnostic_hash_D2H_bytes=sum(x.numel()*x.element_size() for x in (qq,kk,vv))))
            return out
        output=super().dispatch(layer,weighted,q,k,v,**kwargs)
        owners=self.pending[layer][0]
        # Last clean layouts and per-call rows remain bounded by the registered
        # finite case; report physical residency, not only an admission flag.
        admitted_resident=sum(o in self.admitted for o in owners if o is not None)
        self.rows[-1].update(admitted_source_resident_cache_tokens=admitted_resident*self.frame_tokens,
            state='return_'+self.return_scope if self.returning else 'current_scene_only' if self.release_active else 'native',
            selector='causal_return_context_filter',
            admitted_source_visible_tokens=self.admitted_visible_tokens,
            source_beta=self.source_beta,source_weight_applied=applied,
            source_weight_backend='native_FA2_disjoint_LSE_merge' if applied else 'native_FA2',
            return_scope=self.return_scope,external_restore_enabled=self.restore_enabled)
        return output

    def audit(self):
        if self.weight_replay and not self.weight_diagnostics:raise RuntimeError('no visible source for registered weight replay')
        result=super().audit()
        result.update(eligibility_restore_factorial=True,return_scope=self.return_scope,
            selector='causal_return_context_filter',
            source_beta=self.source_beta,weight_diagnostics=self.weight_diagnostics,
            weight_index_H2D_bytes=self.weight_index_H2D_bytes,weight_prepare_host_s=self.weight_prepare_host_s,
            external_restore_enabled=self.restore_enabled,source_residency=self.source_residency,
            admission_is_not_residency=True,metadata_event_capacity=2048,
            no_new_archive_without_recall=not self.restore_enabled,
            scope='current-phase eligibility x actual bounded clean8 archive restoration; versions explicitly admitted')
        return result
