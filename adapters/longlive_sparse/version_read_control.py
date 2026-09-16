"""Fixed old/new read controls over the SAME installed old4+new4 bank.

Each policy is constant across requests. Choosing the compatible arm afterward
is an isolated oracle analysis, not an implemented online intent recognizer.
All eight frames are stored/transferred; actual visible source is four frames.
"""
import torch
import time
from .wave2_temporal_budget import Wave2TemporalBudget,updated_owners,classify_window
from .native_resident_history import window_slot_indices


def permitted_version_positions(owners,physical,mapping,policy):
    if policy not in ('old','new'):raise ValueError('fixed old/new control required')
    if not mapping:return list(range(len(physical)))
    wanted=min(mapping.values()) if policy=='old' else max(mapping.values())
    return [i for i,slot in enumerate(physical) if owners[slot] not in mapping or mapping[owners[slot]]==wanted]


class VersionReadControl(Wave2TemporalBudget):
    def __init__(self,pipe,method,*,version_read='all',version_kv_role=None,**kwargs):
        if method!='w2_full_recall' or kwargs.get('version_policy')!='old4_new4':
            raise ValueError('fixed read control requires the same raw old4+new4 bank')
        super().__init__(pipe,method,**kwargs)
        if version_read not in ('all','old','new'):raise ValueError('unknown fixed version read')
        if version_kv_role not in (None,'oldk_newv','newk_oldv'):raise ValueError('unknown K/V version role')
        if version_kv_role is not None and version_read!='all':raise ValueError('K/V role control owns both versions')
        self.version_kv_role_staged=kwargs.get('version_kv_role_staged')
        if self.version_kv_role_staged not in (None,'oldk_newv','newk_oldv'):raise ValueError('unknown staged K/V version role')
        if self.version_kv_role_staged is not None and (version_kv_role is not None or version_read!='all'):
            raise ValueError('staged K/V role owns both versions and replaces the gather control')
        self.version_read=version_read;self.version_kv_role=version_kv_role;self.version_owners={};self.read_indices={};self.read_index_bytes=0
        self.read_prepare_host_s=0.

    def before(self,owner,values,kwargs):
        prior=len(self.storage_events);super().before(owner,values,kwargs)
        if len(self.storage_events)>prior:
            plan=self.scene.installations[-1]['installation']['admission_plan']
            first,last=self.storage_events[-1]['slot_range']
            self.version_owners=dict(zip(self.owners[0][first:last],plan['source_versions']))
            if sorted(self.version_owners.values()).count(min(self.version_owners.values()))!=4:
                raise RuntimeError('expected four raw frames in each version')

    def dispatch(self,layer,original,q,k,v,**kwargs):
        began=time.perf_counter();new_index_bytes=0
        info=kwargs['info']
        owners=updated_owners(self.owners[layer],info,kwargs['current_start'],self.frame_tokens,self.calls,self.phase)
        physical=window_slot_indices(end=kwargs['cache_end'],start=kwargs['window_start'],effective_sink=kwargs['effective_sink'],
            pinned_start=kwargs['pinned_start'],pinned_len=kwargs['pinned_len'],prepend_sink=kwargs['prepend_sink'],
            prepend_pinned=kwargs['prepend_pinned'],max_tokens=kwargs['max_tokens'],frame_tokens=self.frame_tokens)
        allowed=permitted_version_positions(owners,physical,self.version_owners,self.version_read) if self.version_kv_role is None and self.version_read!='all' else list(range(len(physical)));removed=set(range(len(physical)))-set(allowed)
        roles=classify_window(owners,physical,info,self.frame_tokens,kwargs['effective_sink'],kwargs['global_sink_tokens'],kwargs['pinned_start'],kwargs['pinned_len'])
        if any(roles[i]['current'] for i in removed):raise RuntimeError('version control removed current tokens')
        indices=None
        if removed:
            key=(k.shape[1],tuple(allowed),str(k.device));indices=self.read_indices.get(key)
            if indices is None:
                indices=torch.tensor([t for i in allowed for t in range(i*self.frame_tokens,(i+1)*self.frame_tokens)],device=k.device,dtype=torch.long)
                self.read_indices[key]=indices;self.read_index_bytes+=indices.numel()*indices.element_size()
                new_index_bytes=indices.numel()*indices.element_size()
                if len(self.read_indices)>32:raise RuntimeError('version read geometry bound32 exceeded')
        k_role = v_role = None;staged_applied=None
        if self.version_kv_role_staged is not None and self.version_owners:
            # Role content was staged into the cache at install time; no per-call
            # remap. Report visibility only: full 4+4 visibility == the role graph
            # is exactly the gather control's applied graph; partial visibility
            # reads staged role content instead of the discarded true halves.
            old_pos=[i for i,slot in enumerate(physical) if self.version_owners.get(owners[slot])==min(self.version_owners.values())]
            new_pos=[i for i,slot in enumerate(physical) if self.version_owners.get(owners[slot])==max(self.version_owners.values())]
            staged_applied=len(old_pos)==4 and len(new_pos)==4
            if not staged_applied:self.version_kv_role_skipped=getattr(self,'version_kv_role_skipped',0)+1
        if self.version_kv_role is not None and self.version_owners:
            old_pos=[i for i,slot in enumerate(physical) if self.version_owners.get(owners[slot])==min(self.version_owners.values())]
            new_pos=[i for i,slot in enumerate(physical) if self.version_owners.get(owners[slot])==max(self.version_owners.values())]
            if len(old_pos)!=4 or len(new_pos)!=4:
                # A later rolling window may evict part of the installed bank;
                # applying a partial role would silently change the intervention.
                self.version_kv_role_skipped = getattr(self, 'version_kv_role_skipped', 0) + 1
                old_pos = new_pos = []
            if not old_pos:
                k_role = v_role = None
            else:
                kt=list(range(len(physical)));vt=list(range(len(physical)))
                k_src,v_src=(old_pos,new_pos) if self.version_kv_role=='oldk_newv' else (new_pos,old_pos)
                for dst,src in zip(v_src,k_src):kt[dst]=src
                for dst,src in zip(k_src,v_src):vt[dst]=src
                k_role=torch.tensor([t for i in kt for t in range(i*self.frame_tokens,(i+1)*self.frame_tokens)],device=k.device,dtype=torch.long)
                v_role=torch.tensor([t for i in vt for t in range(i*self.frame_tokens,(i+1)*self.frame_tokens)],device=v.device,dtype=torch.long)
        def filtered(qq,kk,vv):
            if k_role is not None:return original(qq,kk.index_select(1,k_role),vv.index_select(1,v_role))
            return original(qq,kk,vv) if indices is None else original(qq,kk.index_select(1,indices),vv.index_select(1,indices))
        prep=time.perf_counter()-began;self.read_prepare_host_s+=prep
        out=super().dispatch(layer,filtered,q,k,v,**kwargs);row=self.rows[-1];excluded=len(removed)*self.frame_tokens
        row.update(actual_K=k.shape[1]-excluded,logical_pairs=q.shape[1]*q.shape[2]*(k.shape[1]-excluded),
            protected_union_tokens=row['protected_union_tokens']-excluded,recalled_tokens=row['recalled_tokens']-excluded,
            pin_tokens=row['pin_tokens']-sum(roles[i]['pin'] for i in removed)*self.frame_tokens,
            version_read=self.version_read,excluded_installed_source_tokens=excluded,
            version_read_prepare_host_s=prep,index_H2D_bytes=row['index_H2D_bytes']+new_index_bytes,
            visible_installed_source_tokens=sum(owners[physical[i]] in self.version_owners for i in allowed)*self.frame_tokens,
            clean_commit_scope='full permitted version graph' if self.version_kv_role_staged is None else 'staged role graph; discarded true halves not restorable',
            version_kv_role=self.version_kv_role,version_kv_role_staged=self.version_kv_role_staged,
            version_kv_role_applied=k_role is not None if staged_applied is None else staged_applied,
            GPU_gather_output_bytes=2*(k.shape[1]-excluded)*k.shape[2]*k.shape[3]*k.element_size() if indices is not None else 0)
        return out

    def audit(self):
        result=super().audit();result.update(version_read=self.version_read,version_kv_role=self.version_kv_role,version_kv_role_staged=self.version_kv_role_staged,version_kv_role_skipped=getattr(self,'version_kv_role_skipped',0),version_read_index_H2D_bytes=self.read_index_bytes,
            version_read_prepare_host_s=self.read_prepare_host_s,
            version_read_index_GPU_bytes=sum(t.numel()*t.element_size() for t in self.read_indices.values()),
            clean_commit='full permitted version graph, never restores excluded version eligibility' if self.version_kv_role_staged is None else 'staged role graph; skipped/partial-visibility calls read staged role content',
            current_request_intent_inferred=False,read_choice_fixed_across_requests=True,
            all8_raw_frames_stored_and_installed=self.version_kv_role_staged is None,
            role_half_bank_only=self.version_kv_role_staged is not None,read_ceiling_is_not_actual_use=True)
        return result
