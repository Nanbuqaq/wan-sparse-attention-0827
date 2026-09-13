"""Bounded descriptor retention separates retrieval identity from raw residency."""
import hashlib,json,time
from .access_motion_memory import BoundedSceneArchive
from .native_scene_admission import choose_scene,has_revisit_cue
from .native_retimed_memory import NativeRetimedEpisodeMemory


def choose_available(text,prototype,catalog,available_versions,frame,**policy):
    decision=choose_scene(text,prototype,catalog,frame,**policy)
    selected=decision['selected_version']
    if selected is not None and selected not in available_versions:
        decision.update(metadata_selected_version=selected,selected_version=None,reason='selected_payload_evicted_abstain')
    return decision


class PayloadAwareScene(BoundedSceneArchive):
    def __init__(self,pipe,**kwargs):
        super().__init__(pipe,**kwargs);self.catalog=[];self.catalog_evictions=0

    def _archive_last_scene(self,frame):
        super()._archive_last_scene(frame)
        self.catalog.append(self.banks[-1]['descriptor'])
        if len(self.catalog)>64:self.catalog.pop(0);self.catalog_evictions+=1
        if sum(x.condition_prototype.numel()*x.condition_prototype.element_size() for x in self.catalog)>1024**2:
            raise RuntimeError('descriptor payload bound1MiB exceeded')
        phase=self.last_commit['phase']
        self.archives[-1]['derived_from_restored_generation']=any(x['current_phase']==phase for x in self.installations)

    def before(self,owner,values,kwargs,*,current_text):
        if max(len(self.archives),len(self.decisions),len(self.installations))>=2048:raise RuntimeError('finite metadata event bound reached')
        frame=int(kwargs['current_start'])//self.pipe.frame_seq_length
        count=self.counts.get(frame,0);self.counts[frame]=count+1
        if count:return
        for old in sorted(self.counts)[:-2]:del self.counts[old]
        phase=float(self.pipe._dit_model.rope_temporal_offset)
        if self.last_commit is not None and phase!=self.last_commit['phase']:self._archive_last_scene(frame)
        if any(x['current_phase']==phase for x in self.installations):
            self.decisions.append(dict(at_latent=frame,selected_version=None,reason='revisit_already_served_this_phase_no_new_installation',scores=[]));return
        prototype=self._prototype(kwargs['conditional_dict']) if has_revisit_cue(current_text) else None
        started=time.perf_counter()
        decision=choose_available(current_text,prototype,self.catalog,{b['descriptor'].archive_version for b in self.banks},frame,**self.policy)
        self.ledger['selector_CPU_wall_s']+=time.perf_counter()-started
        decision.update(at_latent=frame,current_text_sha256=hashlib.sha256(current_text.encode()).hexdigest())
        self.decisions.append(decision)
        if decision['selected_version'] is None:return
        bank=next(b for b in self.banks if b['descriptor'].archive_version==decision['selected_version']);source=bank['descriptor']
        install=NativeRetimedEpisodeMemory(self.pipe,mode='raw_reveal',source_end=source.source_end,target_start=frame,prompts=[],destination='shot')
        install.bank=bank['kv'];install.capture=dict(completed_latents=source.source_end,source_frames=list(range(source.source_end-8,source.source_end)),source_temporal_offset=source.source_phase,K_storage='already_absolute_RoPE_positioned')
        install._install();plan=install.install['admission_plan']
        plan.update(method='causal_cue_T5_latest_scene_history',archive_version=source.archive_version,selection_policy=self.policy)
        sha=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        install.install.update(admission_plan_sha256=sha,KV_storage_version_sha256=sha)
        self.installations.append(dict(at_latent=frame,current_phase=phase,installation=install.install,transfer_ledger=install.ledger))
        self.ledger['history_H2D_KV_bytes']+=install.ledger['demand_H2D_payload_bytes']
        self.ledger['history_install_wall_s']+=install.ledger['demand_wall_s']
        self.ledger['temporal_rebind_wall_s']+=install.ledger['temporal_key_rebind_wall_s']

    def audit(self):
        result=super().audit();live={b['descriptor'].archive_version for b in self.banks}
        result.update(payload_aware_catalog=True,catalog_capacity=64,catalog_evictions=self.catalog_evictions,
            catalog_prototype_bytes=sum(x.condition_prototype.numel()*x.condition_prototype.element_size() for x in self.catalog),
            catalog=[dict(version=x.archive_version,source_end=x.source_end,payload_available=x.archive_version in live) for x in self.catalog],
            scope='abstain if best known descriptor has no raw payload; not a general entity resolver or infinite catalog')
        return result
