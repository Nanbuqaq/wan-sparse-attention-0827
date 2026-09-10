"""Explicit temporal rebinding of selected past keys; never a layout-only change."""
import hashlib
import json
import time

import torch

from .native_episode_memory import NativeEpisodeMemory
from .native_temporal_rephase import rephase_temporal_keys


def binding_coordinates(*,source_frame,target_frame,frames,source_phase,current_phase,policy):
    """Separate two contributions to the SAME temporal rotation, not new axes."""
    if policy not in ('recent_virtual','phase_only','age_only'):
        raise ValueError('unknown temporal binding policy')
    if frames < 1 or source_frame < 0 or source_frame+frames > target_frame:
        raise ValueError('only committed, strictly historical blocks may be rebound')
    effective_frame=source_frame if policy=='phase_only' else target_frame-frames
    effective_phase=source_phase if policy=='age_only' else current_phase
    age_delta=effective_frame-source_frame
    phase_delta=effective_phase-source_phase
    return dict(virtual_source_frames=list(range(effective_frame,effective_frame+frames)),
        bound_phase=float(effective_phase),age_delta=age_delta,phase_delta=float(phase_delta),
        temporal_delta=float(age_delta+phase_delta))


class NativeRetimedEpisodeMemory(NativeEpisodeMemory):
    def __init__(self,pipe,*,position_policy='recent_virtual',**kwargs):
        if position_policy not in ('recent_virtual','phase_only','age_only'):
            raise ValueError('unknown temporal binding policy')
        self.position_policy=position_policy
        if kwargs.get('mode') not in ('raw_reveal','raw_away') or kwargs.get('destination')!='shot' or kwargs.get('restore_after_frames',0):
            raise ValueError('retiming requires one raw source in shot slots without TTL')
        if pipe._dit_model.t_scale!=1 or pipe._dit_model.rope_method!='linear' or pipe._dit_model.original_seq_len is not None:
            raise ValueError('retiming is qualified only for native linear scale1 RoPE')
        super().__init__(pipe,**kwargs)

    def _install(self):
        super()._install()
        plan=self.install['admission_plan'];start,end=plan['destination_token_range']
        source_frame=self.source_end-self.frames
        old_offset=float(self.capture['source_temporal_offset']);new_offset=float(self.pipe._dit_model.rope_temporal_offset)
        coordinates=binding_coordinates(source_frame=source_frame,target_frame=self.target,frames=self.frames,
            source_phase=old_offset,current_phase=new_offset,policy=self.position_policy)
        delta=coordinates['temporal_delta']
        torch.cuda.synchronize();began=time.perf_counter();written=0
        for cache in self.pipe.kv_cache_pos:
            key=cache['k'][:,start:end]
            key.copy_(rephase_temporal_keys(key,delta))
            written+=key.numel()*key.element_size()
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-began
        self.ledger['temporal_key_rebind_wall_s']=elapsed
        self.ledger['temporal_key_rebind_GPU_output_bytes']=written
        self.ledger['demand_wall_s']+=elapsed
        name=('archived_temporal_K_rebound_to_recent_virtual_history' if self.position_policy=='recent_virtual'
              else 'archived_temporal_K_'+self.position_policy)
        plan.update(position_policy=name,binding_policy=self.position_policy,
            actual_source_frames=plan['source_frames'],**coordinates,
            source_phase=old_offset,current_phase=new_offset,
            spatial_key_channels_unchanged=True,value_unchanged=True,preRoPE_raw_K_recovery_claimed=False)
        sha=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.install.update(admission_plan_sha256=sha,KV_storage_version_sha256=sha,original_K_preserved=False)

    def audit(self):
        report=super().audit()
        report.update(K_positions_preserved_not_rebased=False,actual_history_frame_IDs_preserved=True,
            temporal_rebinding_is_algorithm_not_layout=True,spatial_K_and_all_V_unchanged=True)
        return report
