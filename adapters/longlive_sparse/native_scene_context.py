"""Privileged scene-switch diagnostic: retrieve one episode and retire away KV.

This changes the logical context, not merely physical layout. It keeps the
original global8, installs the committed source8 in the shot role, and makes
the previous-scene suffix inaccessible before the first return denoising call.
"""
import hashlib
import json
import time

import torch

from .native_commit_replay import cache_metadata
from .native_episode_memory import NativeEpisodeMemory


class NativeSceneContextReset(NativeEpisodeMemory):
    def __init__(self,pipe,**kwargs):
        if kwargs.get('mode') not in ('raw_reveal','raw_away') or kwargs.get('destination')!='shot' or kwargs.get('restore_after_frames',0):
            raise ValueError('scene reset requires raw relevant/wrong episode in shot role without TTL')
        super().__init__(pipe,**kwargs)

    def _install(self):
        primary=self.pipe.kv_cache_pos;n=self.frames*self.pipe.frame_seq_length
        before=cache_metadata(primary)
        if before['pinned_start']!=n or before['pinned_len']!=n or before['local_end_index']<2*n:
            raise RuntimeError('frozen reset requires global8 followed by previous-shot8')
        super()._install()
        torch.cuda.synchronize();started=time.perf_counter()
        for c in primary:
            c['local_end_index'].fill_(2*n)
            # pinned_start/pinned_len and global absolute clock stay unchanged.
        torch.cuda.synchronize()
        after=cache_metadata(primary)
        expected=dict(before,local_end_index=2*n)
        if after!=expected:raise RuntimeError('unexpected context reset metadata mutation')
        self.ledger['context_reset_wall_s']=time.perf_counter()-started
        self.ledger['context_reset_invalidated_tokens_per_head_per_layer']=before['local_end_index']-2*n
        plan=self.install['admission_plan']
        plan.update(method='privileged_native_scene_context_reset',
            active_history_before_return_tokens_per_head=2*n,
            preserve_original_global=True,retire_previous_scene_suffix_before_denoising=True)
        self.install.update(admission_plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            cache_metadata_unchanged=False,cache_metadata_transition=dict(before=before,after=after),
            native_attention_size_unchanged=False,native_allocated_capacity_unchanged=True,
            logical_context_changed=True,unused_suffix_excluded_by_native_endpoint_not_zero_attention=True)

    def audit(self):
        result=super().audit()
        result.update(scene_context_reset=True,logical_context_change_not_layout_optimization=True,
                      preserves_original_global_and_absolute_clock=True)
        return result
