"""One observed state update: fixed raw8 capacity and per-frame temporal binding."""
import hashlib,json,time
import torch
from .native_causal_scene_memory import NativeCausalSceneMemory
from .native_commit_replay import cache_metadata
from .native_scene_admission import SceneDescriptor,choose_scene,has_revisit_cue
from .native_temporal_rephase import rephase_temporal_keys
from .version_frame_bank import EightFrameBank


def frame_binding(records,target,current_phase):
    if len(records)!=8 or any(r['frame']>=target for r in records):raise ValueError('strictly past raw8 required')
    return [float(target-8+i+current_phase-r['frame']-r['phase']) for i,r in enumerate(records)]


class VersionSceneMemory(NativeCausalSceneMemory):
    def __init__(self,pipe,*,version_policy):
        super().__init__(pipe)
        if version_policy not in ('latest8','old4_new4','uniform8'):raise ValueError('unknown version policy')
        self.version_policy=version_policy;self.frame_bank=None;self.identity_anchor=None
        self.rejected_writes=[]

    def _archive_last_scene(self,current_frame):
        last=self.last_commit
        if last is None or last['end']!=current_frame:raise RuntimeError('source must be just committed')
        if len(self.archives)+len(self.rejected_writes)>=128:raise RuntimeError('finite version diagnostic metadata cap')
        prototype=last['prototype']
        similarity=1. if self.identity_anchor is None else float(torch.dot(prototype,self.identity_anchor))
        if similarity<.8 or (self.frame_bank is not None and self.frame_bank.updates==2):
            self.rejected_writes.append(dict(frame=current_frame,similarity=similarity,
                reason='different_text_descriptor' if similarity<.8 else 'one_update_diagnostic_complete'))
            return
        began=time.perf_counter();torch.cuda.synchronize();caches=self.pipe.kv_cache_pos;ft=self.pipe.frame_seq_length
        if any(int(c['global_end_index'])!=current_frame*ft or int(c['local_end_index'])<8*ft for c in caches):
            raise RuntimeError('version source bounds differ')
        if self.frame_bank is None:
            cap=sum(8*ft*c['k'].shape[0]*c['k'].shape[2]*c['k'].shape[3]*(c['k'].element_size()+c['v'].element_size()) for c in caches)
            self.frame_bank=EightFrameBank(ft,cap);self.identity_anchor=prototype.clone()
        self.version+=1
        records=[dict(frame=f,phase=last['phase'],version=self.version) for f in range(current_frame-8,current_frame)]
        self.frame_bank.update(caches,records,self.version_policy);torch.cuda.synchronize()
        descriptor=SceneDescriptor(self.version,current_frame,last['phase'],prototype)
        self.banks=[dict(descriptor=descriptor,kv=self.frame_bank.kv,owned_bytes=self.frame_bank.raw_bytes)]
        self.archives.append(dict(archive_version=self.version,source_end=current_frame,source_phase=last['phase'],
            source_frames=[r['frame'] for r in self.frame_bank.records],bank=self.frame_bank.audit(),text_similarity_to_v0=similarity))
        self.ledger['archive_D2H_KV_bytes']=self.frame_bank.D2H_bytes
        self.ledger['CPU_archive_peak_tensor_bytes']=self.frame_bank.raw_bytes
        self.ledger['archive_wall_s']+=time.perf_counter()-began

    def before(self,owner,values,kwargs,*,current_text):
        frame=int(kwargs['current_start'])//self.pipe.frame_seq_length
        count=self.counts.get(frame,0);self.counts[frame]=count+1
        if count:return
        if len(self.counts)>2:
            for old in sorted(self.counts)[:-2]:del self.counts[old]
        phase=float(self.pipe._dit_model.rope_temporal_offset)
        if self.last_commit is not None and phase!=self.last_commit['phase']:self._archive_last_scene(frame)
        if any(r['current_phase']==phase for r in self.installations):return
        prototype=self._prototype(kwargs['conditional_dict']) if has_revisit_cue(current_text) else None
        decision=choose_scene(current_text,prototype,[b['descriptor'] for b in self.banks],frame,**self.policy)
        decision.update(at_latent=frame);self.decisions.append(decision)
        if decision['selected_version'] is None:return
        if self.frame_bank.updates!=2:raise RuntimeError('version experiment requires two actual admitted source writes')
        self._install_frames(frame,phase)

    def _install_frames(self,frame,phase):
        pipe=self.pipe;ft=pipe.frame_seq_length;caches=pipe.kv_cache_pos
        if pipe._dit_model.t_scale!=1 or pipe._dit_model.rope_method!='linear' or pipe._dit_model.original_seq_len is not None:
            raise ValueError('native linear scale1 temporal binding only')
        before=cache_metadata(caches);start=before['pinned_start'];n=8*ft
        if before['pinned_len']!=n or start<n or start+n>before['local_end_index']:
            raise RuntimeError('distinct native shot8 installation region required')
        slots=self.frame_bank.ordered_slots();records=[self.frame_bank.records[i] for i in slots]
        deltas=frame_binding(records,frame,phase);moved=0;rephase_s=0.
        torch.cuda.synchronize();began=time.perf_counter()
        for cache,(key,value) in zip(caches,self.frame_bank.kv):
            for index,slot in enumerate(slots):
                dst=slice(start+index*ft,start+(index+1)*ft);src=slice(slot*ft,(slot+1)*ft)
                cache['k'][:,dst].copy_(key[:,src]);cache['v'][:,dst].copy_(value[:,src])
                moved+=key[:,src].numel()*key.element_size()+value[:,src].numel()*value.element_size()
                t=time.perf_counter();cache['k'][:,dst].copy_(rephase_temporal_keys(cache['k'][:,dst],deltas[index]))
                rephase_s+=time.perf_counter()-t
        torch.cuda.synchronize();elapsed=time.perf_counter()-began
        if cache_metadata(caches)!=before:raise RuntimeError('version install changed native metadata')
        plan=dict(archive_version=self.version,actual_source_frames=[r['frame'] for r in records],
            source_frames=[r['frame'] for r in records],source_phases=[r['phase'] for r in records],
            source_versions=[r['version'] for r in records],source_phase=self.last_source_phase(),
            virtual_source_frames=list(range(frame-8,frame)),bound_phase=phase,current_phase=phase,
            destination_token_range=[start,start+n],version_policy=self.version_policy,
            temporal_deltas=deltas,position_policy='each source frame rebound to ordered recent8; not layout-only')
        digest=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
        install=dict(admission_plan=plan,admission_plan_sha256=digest,KV_storage_version_sha256=digest)
        self.installations.append(dict(at_latent=frame,current_phase=phase,installation=install))
        self.ledger['history_H2D_KV_bytes']+=moved;self.ledger['history_install_wall_s']+=elapsed
        self.ledger['temporal_rebind_wall_s']=None
        self.ledger['temporal_rebind_host_enqueue_s']=self.ledger.get('temporal_rebind_host_enqueue_s',0.)+rephase_s

    def last_source_phase(self):return self.banks[0]['descriptor'].source_phase

    def audit(self):
        result=super().audit();result.update(version_policy=self.version_policy,
            archive_budget_bytes=self.frame_bank.capacity_bytes if self.frame_bank else 0,
            install_cost_scope='complete synchronized H2D+rephase wall; rephase enqueue is not isolated GPU time',
            frame_bank=self.frame_bank.audit() if self.frame_bank else None,rejected_writes=self.rejected_writes,
            descriptor_CPU_bytes=sum(t.numel()*t.element_size() for t in
                ([self.identity_anchor,self.banks[0]['descriptor'].condition_prototype] if self.banks else [])),
            scope='one observed update, initial scene text anchor, frozen .8 write similarity; not entity/state disentanglement',
            uniform_scope='uniform8 over the two clean8 versions only; not whole-history uniform oracle')
        return result
