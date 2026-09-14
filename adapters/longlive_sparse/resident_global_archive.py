"""Elide an unusable cold copy while preserving descriptor/FIFO semantics."""
import time
from .immutable_source_reader import SideArchive
from .native_scene_admission import SceneDescriptor


class ResidentGlobalArchive(SideArchive):
    def __init__(self,pipe,**kwargs):
        super().__init__(pipe,**kwargs)
        self.actual_peak=0;self.virtual_peak=0;self.elided_bytes=0

    def _actual_peak(self):
        self.actual_peak=max(self.actual_peak,sum(b.get('physical_owned_bytes',b['owned_bytes']) for b in self.banks))
        self.virtual_peak=max(self.virtual_peak,sum(b['owned_bytes'] for b in self.banks))
        self.ledger['CPU_archive_peak_tensor_bytes']=self.actual_peak

    def _archive_last_scene(self,current_frame):
        if current_frame!=8:
            super()._archive_last_scene(current_frame);self._actual_peak();return
        began=time.perf_counter();last=self.last_commit;n=8*self.pipe.frame_seq_length
        if last is None or last['end']!=8:raise RuntimeError('initial global snapshot is not committed')
        caches=self.pipe.kv_cache_pos
        if any(int(c['global_end_index'])!=n or int(c['local_end_index'])!=n for c in caches):
            raise RuntimeError('elision requires the intact initial native global8')
        required=sum(n*c['k'].shape[0]*c['k'].shape[2]*c['k'].shape[3]*(c['k'].element_size()+c['v'].element_size()) for c in caches)
        metadata=last['prototype'].numel()*last['prototype'].element_size();virtual=required+metadata
        if virtual>self.budget:raise RuntimeError('reference logical archive would exceed the budget')
        while sum(b['owned_bytes'] for b in self.banks)+virtual>self.budget:
            self.banks.pop(0);self.ledger['evicted_archives']+=1
        self.version+=1;descriptor=SceneDescriptor(self.version,8,last['phase'],last['prototype'])
        # The original reader rejects an external restore before touching kv
        # whenever these frames remain native-resident. Preserve that behavior.
        self.banks.append(dict(descriptor=descriptor,kv=None,owned_bytes=virtual,
            physical_owned_bytes=metadata,payload_kind='resident_initial_global'))
        self.archives.append(dict(archive_version=self.version,source_frames=list(range(8)),source_end=8,
            source_phase=last['phase'],KV_bytes=0,condition_prototype_bytes=metadata,
            reference_KV_bytes=required,virtual_budget_bytes=virtual,
            payload_kind='resident_initial_global',ready=True))
        self.elided_bytes+=required;self._actual_peak()
        self.ledger['archive_wall_s']+=time.perf_counter()-began

    def audit(self):
        result=super().audit()
        result['resident_global_elision']=dict(elided_D2H_KV_bytes=self.elided_bytes,
            actual_CPU_archive_peak_bytes=self.actual_peak,virtual_reference_budget_peak_bytes=self.virtual_peak,
            reference_descriptor_and_eviction_sequence_preserved=True,extra_archive_capacity_claimed=False,
            resident_source_external_restore_still_rejected=True,
            scope='only source0-7, permanently present in the qualified immutable native global cache')
        return result
