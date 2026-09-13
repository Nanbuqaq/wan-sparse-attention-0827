"""Causal scene-release diagnostic: filter old scene KV, retain an independent CPU copy."""
import time
import resource
import torch
from .wave2_temporal_budget import Wave2TemporalBudget
from .native_resident_history import window_slot_indices
from .wave2_temporal_budget import updated_owners,classify_window
from .native_scene_admission import has_revisit_cue


def scene_positions(owners,physical,phase,active):
    if any(owners[s] is None or owners[s][0]!='native' for s in physical):
        raise ValueError('scene diagnostic requires actual native ownership')
    return [i for i,s in enumerate(physical) if not active or owners[s][3]==phase]


class NativeSceneRelease(Wave2TemporalBudget):
    def __init__(self,pipe,method,**kwargs):
        if method!='w2_scene_release':raise ValueError('explicit scene diagnostic method required')
        super().__init__(pipe,method,**kwargs)
        if self.observer or self.capture_enabled:raise ValueError('no additional tensor observer in initial scene diagnostic')
        self.release_active=False;self.release_phase=None;self.retired={};self.retired_bytes=0
        self.retired_budget=16*1024**3;self.retire_host_s=0.;self.release_events=[];self.release_indices={}

    def archive_resident(self):
        work=[];needed=0
        for layer,owners in enumerate(self.owners):
            cache=self.pipe.kv_cache_pos[layer]
            for slot,owner in enumerate(owners):
                key=(layer,owner)
                if owner is None or key in self.retired:continue
                if owner[1]>=self.active_start//self.frame_tokens:raise RuntimeError('cannot retire future/uncommitted KV')
                start=slot*self.frame_tokens;stop=start+self.frame_tokens
                k=cache['k'][:,start:stop];v=cache['v'][:,start:stop]
                size=k.numel()*k.element_size()+v.numel()*v.element_size();needed+=size;work.append((key,k,v,size))
        if self.retired_bytes+needed>self.retired_budget:raise RuntimeError('scene-release independent archive exceeds16GiB')
        started=time.perf_counter()
        for key,k,v,size in work:
            self.retired[key]=(k.detach().to('cpu',copy=True),v.detach().to('cpu',copy=True));self.retired_bytes+=size
        self.retire_host_s+=time.perf_counter()-started
        return needed

    def before(self,owner,values,kwargs):
        previous=self.last_phase;super().before(owner,values,kwargs)
        if previous is not None and previous!=self.phase:
            frame=self.active_start//self.frame_tokens;text=self.current_text(frame);return_cue=has_revisit_cue(text)
            copied=0
            if return_cue:self.release_active=False
            else:
                copied=self.archive_resident();self.release_active=True;self.release_phase=self.phase
            self.release_events.append(dict(frame=frame,phase=self.phase,current_text_return_cue=return_cue,
                release_active=self.release_active,new_archive_D2H_bytes=copied,
                archive_restored=False,decision_inputs='current phase/current raw text + past actual owner tuples'))

    def dispatch(self,layer,original,q,k,v,**kwargs):
        began=time.perf_counter();info=kwargs['info']
        owners=updated_owners(self.owners[layer],info,kwargs['current_start'],self.frame_tokens,self.calls,self.phase)
        physical=window_slot_indices(end=kwargs['cache_end'],start=kwargs['window_start'],effective_sink=kwargs['effective_sink'],
            pinned_start=kwargs['pinned_start'],pinned_len=kwargs['pinned_len'],prepend_sink=kwargs['prepend_sink'],
            prepend_pinned=kwargs['prepend_pinned'],max_tokens=kwargs['max_tokens'],frame_tokens=self.frame_tokens)
        roles=classify_window(owners,physical,info,self.frame_tokens,kwargs['effective_sink'],kwargs['global_sink_tokens'],kwargs['pinned_start'],kwargs['pinned_len'])
        allowed=scene_positions(owners,physical,self.release_phase,self.release_active);kept=set(allowed)
        if any(role['current'] and i not in kept for i,role in enumerate(roles)):raise RuntimeError('current context was filtered')
        indices=None;index_bytes=0
        if len(allowed)!=len(physical):
            key=(k.shape[1],tuple(allowed),str(k.device));indices=self.release_indices.get(key)
            if indices is None:
                indices=torch.tensor([t for i in allowed for t in range(i*self.frame_tokens,(i+1)*self.frame_tokens)],device=k.device,dtype=torch.long)
                self.release_indices[key]=indices;index_bytes=indices.numel()*indices.element_size()
            if len(self.release_indices)>32:raise RuntimeError('scene geometry cache exceeded registered32 layouts')
        prep=time.perf_counter()-began
        def filtered(qq,kk,vv):
            return original(qq,kk,vv) if indices is None else original(qq,kk.index_select(1,indices),vv.index_select(1,indices))
        output=super().dispatch(layer,filtered,q,k,v,**kwargs);row=self.rows[-1];actual=len(allowed)*self.frame_tokens
        protected=sum(any(role.values()) for i,role in enumerate(roles) if i in kept)*self.frame_tokens
        row.update(state='current_scene_only' if indices is not None else 'native',actual_K=actual,
            logical_pairs=q.shape[1]*q.shape[2]*actual,native_protected_union_tokens=row['protected_union_tokens'],
            protected_union_tokens=protected,optional_tokens=actual-protected,selected_optional_tokens=actual-protected,
            native_sink_tokens=row['sink_tokens'],native_pin_tokens=row['pin_tokens'],
            sink_tokens=sum(role['sink'] for i,role in enumerate(roles) if i in kept)*self.frame_tokens,
            pin_tokens=sum(role['pin'] for i,role in enumerate(roles) if i in kept)*self.frame_tokens,
            intentionally_excluded_old_scene_tokens=k.shape[1]-actual,scene_release_prepare_host_s=prep,
            index_H2D_bytes=index_bytes,GPU_gather_output_bytes=0 if indices is None else 2*actual*k.shape[2]*k.shape[3]*k.element_size(),
            selector='current_scene_owner_filter',clean_commit_scope='full permitted current-scene graph, not original full graph')
        return output

    def audit(self):
        result=super().audit();result.update(scene_release_events=self.release_events,CPU_retired_KV_bytes=self.retired_bytes,
            CPU_retired_KV_budget_bytes=self.retired_budget,retired_archive_D2H_bytes=self.retired_bytes,retire_host_including_readiness_s=self.retire_host_s,
            process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,retired_owner_records=len(self.retired),
            retired_archive_restored=False,old_GPU_KV_not_zeroed_or_rephased=True,optional_fraction=None,
            selector='current_scene_owner_filter',no_new_archive_without_recall=False,
            cut_first_chunk='full permitted current-scene graph on nonreturn cuts',recalled_chunk=None,
            clean_commit='full permitted current-scene graph while release active',
            scope='explicit scene-control diagnostic; archive retained independently but not restored; no same-budget or complete-memory-success claim')
        return result
