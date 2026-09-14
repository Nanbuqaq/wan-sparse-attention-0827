"""Use actual retained same-scene history instead of assuming latest is stable."""
import time
import torch

from .immutable_source_reader import SideArchive,request_key
from .native_commit_replay import owned_cpu
from .native_scene_admission import SceneDescriptor,choose_scene,has_revisit_cue


def oldest_same_phase_window(owners,phase,frames=8):
    valid=sorted((o[1],slot) for slot,o in enumerate(owners) if o is not None and o[0]=='native' and o[3]==phase)
    if len(valid)<frames:raise ValueError('not enough actually resident same-phase source frames')
    selected=valid[:frames];ids=[x[0] for x in selected];slots=[x[1] for x in selected]
    if ids!=list(range(ids[0],ids[0]+frames)) or slots!=list(range(slots[0],slots[0]+frames)):
        raise ValueError('registered oldest resident8 must be temporally and physically contiguous')
    return ids,slots


class ResidentSnapshotArchive(SideArchive):
    def __init__(self,pipe,*,owner_provider,window,**kwargs):
        if window not in ('latest8','oldest_resident8'):raise ValueError('unknown source snapshot window')
        super().__init__(pipe,**kwargs);self.owner_provider=owner_provider;self.window=window

    def _archive_last_scene(self,current_frame):
        if self.window=='latest8':
            super()._archive_last_scene(current_frame)
            self.banks[-1]['scene_closed_at']=current_frame
            self.archives[-1].update(scene_closed_at=current_frame,snapshot_window=self.window)
            return
        last=self.last_commit
        if last is None or last['end']!=current_frame:raise RuntimeError('only a just-closed committed scene may be captured')
        geometry=[oldest_same_phase_window(owners,last['phase']) for owners in self.owner_provider()]
        if len({tuple(x[0]) for x in geometry})!=1:raise RuntimeError('source frame IDs differ across layers')
        source_frames=geometry[0][0];n=8*self.pipe.frame_seq_length;caches=self.pipe.kv_cache_pos
        required=sum(n*c['k'].shape[0]*c['k'].shape[2]*c['k'].shape[3]*(c['k'].element_size()+c['v'].element_size()) for c in caches)
        owned=required+last['prototype'].numel()*last['prototype'].element_size()
        if owned>self.budget:raise RuntimeError('one snapshot exceeds actual CPU budget')
        began=time.perf_counter();torch.cuda.synchronize()
        while sum(b['owned_bytes'] for b in self.banks)+owned>self.budget:
            self.banks.pop(0);self.ledger['evicted_archives']+=1
        kv=[]
        for cache,(_,slots) in zip(caches,geometry):
            if int(cache['global_end_index'])!=current_frame*self.pipe.frame_seq_length:raise RuntimeError('source cache not committed at scene close')
            start=slots[0]*self.pipe.frame_seq_length
            kv.append((owned_cpu(cache['k'][:,start:start+n]),owned_cpu(cache['v'][:,start:start+n])))
        storage={t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for pair in kv for t in pair}
        if sum(storage.values())!=required:raise RuntimeError('source snapshot does not own compact raw storage')
        self.version+=1;actual_end=source_frames[-1]+1
        descriptor=SceneDescriptor(self.version,actual_end,last['phase'],last['prototype'])
        self.banks.append(dict(descriptor=descriptor,kv=kv,owned_bytes=owned,scene_closed_at=current_frame))
        self.archives.append(dict(archive_version=self.version,source_frames=source_frames,source_end=actual_end,
            scene_closed_at=current_frame,source_phase=last['phase'],KV_bytes=required,condition_prototype_bytes=owned-required,
            snapshot_window=self.window,unique_owned_raw_storage_bytes=required,logical_tensor_bytes=required,
            storage_count=len(storage),ready=True,source_was_actually_native_resident_at_write=True))
        self.ledger['archive_D2H_KV_bytes']+=required
        self.ledger['CPU_archive_peak_tensor_bytes']=max(self.ledger['CPU_archive_peak_tensor_bytes'],sum(b['owned_bytes'] for b in self.banks))
        self.ledger['archive_wall_s']+=time.perf_counter()-began

    def before(self,owner,values,kwargs,*,current_text,allow_select=True):
        frame=int(kwargs['current_start'])//self.pipe.frame_seq_length
        count=self.counts.get(frame,0);self.counts[frame]=count+1
        if count:return None
        for old in sorted(self.counts)[:-2]:del self.counts[old]
        if max(len(self.archives),len(self.decisions))>=2048:raise RuntimeError('finite source event bound reached')
        phase=float(self.pipe._dit_model.rope_temporal_offset)
        if self.last_commit is not None and self.last_commit['phase']!=phase:self._archive_last_scene(frame)
        if not allow_select:return None
        prototype=self._prototype(kwargs['conditional_dict']) if has_revisit_cue(current_text) else None
        # Preserve the original scene-close cooldown/ranking clock in both
        # arms. Actual source positions remain separate and drive K rebinding.
        descriptors=[SceneDescriptor(b['descriptor'].archive_version,b['scene_closed_at'],b['descriptor'].source_phase,
                     b['descriptor'].condition_prototype) for b in self.banks]
        began=time.perf_counter();decision=choose_scene(current_text,prototype,descriptors,frame,**self.policy)
        self.ledger['selector_CPU_wall_s']+=time.perf_counter()-began
        for score in decision['scores']:
            score['scene_closed_at']=score.pop('source_end')
            score['actual_source_end']=next(b['descriptor'].source_end for b in self.banks if b['descriptor'].archive_version==score['archive_version'])
        decision.update(at_latent=frame,current_request_key=request_key(current_text),selection_clock='scene_closed_at')
        self.decisions.append(decision)
        return next((b for b in self.banks if b['descriptor'].archive_version==decision['selected_version']),None)

    def audit(self):
        result=super().audit()
        result.update(snapshot_window=self.window,selection_clock='scene_closed_at',raw_source_clock='actual source frame IDs',
            snapshot_state_not_inferred_from_prompt=True,no_extra_pending_raw_pool=True,
            routing_descriptor_scope='last requested scene text, held fixed across snapshot controls; not observed state or necessarily exact snapshot conditioning')
        return result
