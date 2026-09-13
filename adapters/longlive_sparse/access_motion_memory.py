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
    def __init__(self,pipe,method,*,restore=False,return_scope='broad',**kwargs):
        if return_scope not in ('broad','narrow'):raise ValueError('unknown return eligibility')
        super().__init__(pipe,method,retired_copy=False,**kwargs)
        self.scene=BoundedSceneArchive(pipe) if restore else None
        self.restore_enabled=restore;self.return_scope=return_scope;self.admitted=set()
        self.source_residency=[];self.returning=False

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

    def allowed_positions(self,owners,physical):
        admitted=self.admitted if self.returning else set()
        selected=eligible_positions(owners,physical,self.release_phase,self.release_active,admitted)
        if len(self.source_residency)>2048:raise RuntimeError('source audit capacity2048 reached')
        return selected

    def audit(self):
        result=super().audit()
        result.update(eligibility_restore_factorial=True,return_scope=self.return_scope,
            external_restore_enabled=self.restore_enabled,source_residency=self.source_residency,
            admission_is_not_residency=True,metadata_event_capacity=2048,
            no_new_archive_without_recall=not self.restore_enabled,
            scope='current-phase eligibility x actual bounded clean8 archive restoration; versions explicitly admitted')
        return result
