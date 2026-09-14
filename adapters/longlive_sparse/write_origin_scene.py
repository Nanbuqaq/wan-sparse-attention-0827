"""Explicit raw-write controls on the existing causal lineage catalog.

Root FIFO replacement is the simple capacity control. Skipping return-derived
snapshots is an anchor diagnostic, not a visual verifier or a state-update rule.
"""
from .payload_aware_scene import PayloadAwareScene


class WriteOriginScene(PayloadAwareScene):
    def __init__(self,pipe,*,write_policy,**kwargs):
        if write_policy not in ('root_latest_fifo','skip_derived'):raise ValueError('unknown write-origin control')
        super().__init__(pipe,**kwargs)
        self.write_policy=write_policy;self.write_events=[];self.root_replacements=0

    def _archive_last_scene(self,frame):
        phase=self.last_commit['phase']
        parent=next((x['installation']['admission_plan']['archive_version'] for x in self.installations
                     if x['current_phase']==phase),None)
        root=self.identity_roots.get(parent,parent) if parent is not None else None
        if self.write_policy=='skip_derived' and parent is not None:
            self.write_events.append(dict(at_latent=frame,phase=phase,action='skip_return_derived_snapshot',
                parent_archive_version=parent,identity_root=root,raw_KV_bytes_written=0,
                new_state_or_view_may_be_lost=True))
            return
        position=None;removed_version=None;removed_bytes=0
        if self.write_policy=='root_latest_fifo' and root is not None:
            position=next((i for i,b in enumerate(self.banks)
                if self.identity_roots.get(b['descriptor'].archive_version,b['descriptor'].archive_version)==root),None)
            if position is not None:
                removed_version=self.banks[position]['descriptor'].archive_version
                removed_bytes=self.banks[position]['owned_bytes']
                # No local payload alias survives this removal. The old raw
                # allocation is released before requesting the replacement.
                self.banks.pop(position);self.root_replacements+=1
        super()._archive_last_scene(frame)
        version=self.banks[-1]['descriptor'].archive_version
        if position is not None:
            self.banks.insert(position,self.banks.pop())
        self.write_events.append(dict(at_latent=frame,phase=phase,
            action='replace_root_preserve_FIFO_position' if position is not None else 'append_new_snapshot',
            parent_archive_version=parent,identity_root=self.identity_roots[version],archive_version=version,
            removed_payload_version=removed_version,removed_owned_bytes_before_new_allocation=removed_bytes,
            raw_KV_bytes_written=self.archives[-1]['KV_bytes']))

    def audit(self):
        result=super().audit()
        result.update(write_policy=self.write_policy,write_events=self.write_events,root_replacements=self.root_replacements,
            root_FIFO_is_first_root_insertion_not_LRU=True,visual_quality_used_for_writes=False,
            skip_derived_is_not_general_state_compatibility=True,
            skipped_writes_create_no_fresh_descriptor_or_fake_new_source_time=True)
        return result
