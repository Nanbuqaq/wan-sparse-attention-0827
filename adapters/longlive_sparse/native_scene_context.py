"""Privileged scene-switch diagnostic: retrieve one episode and retire away KV.

This changes the logical context, not merely physical layout. It keeps the
original global8, installs the committed source8 in the shot role, and makes
the previous-scene suffix inaccessible before the first return denoising call.
"""
import hashlib
import json
import time
from collections import Counter
from unittest.mock import patch

import torch

from .native_commit_replay import cache_metadata
from .native_episode_memory import NativeEpisodeMemory


class NativeSceneContextReset(NativeEpisodeMemory):
    def __init__(self,pipe,*,anchor_policy='keep',**kwargs):
        if anchor_policy not in ('keep','source_only','source_repeat'):raise ValueError('unknown initial anchor policy')
        expected_destination='shot' if anchor_policy=='keep' else 'global'
        if kwargs.get('mode') not in ('raw_reveal','raw_away') or kwargs.get('destination')!=expected_destination or kwargs.get('restore_after_frames',0):
            raise ValueError('scene reset requires raw relevant/wrong episode, the policy-specific destination, and no TTL')
        super().__init__(pipe,**kwargs)
        self.anchor_policy=anchor_policy
        self.active_start=None;self.attention_shapes=Counter();self.attention_patch=None

    def before(self,owner,values,kwargs):
        self.active_start=int(kwargs['current_start'])
        super().before(owner,values,kwargs)

    def observe_attention(self,original,q,k,v,*args,**kwargs):
        # Tensor shape metadata only: no D2H, reductions, or parameter access.
        if self.active_start is not None and self.active_start>=self.target*self.pipe.frame_seq_length:
            self.attention_shapes[(self.active_start//self.pipe.frame_seq_length,int(q.shape[1]),int(k.shape[1]))]+=1
        return original(q,k,v,*args,**kwargs)

    def attach(self):
        import wan_5b.modules.causal_model as native
        original=native.attention
        self.attention_patch=patch.object(native,'attention',lambda q,k,v,*a,**kw:self.observe_attention(original,q,k,v,*a,**kw))
        self.attention_patch.start()
        super().attach()

    def detach(self):
        super().detach()
        if self.attention_patch is not None:self.attention_patch.stop();self.attention_patch=None

    def _install(self):
        primary=self.pipe.kv_cache_pos;n=self.frames*self.pipe.frame_seq_length
        before=cache_metadata(primary)
        if before['pinned_start']!=n or before['pinned_len']!=n or before['local_end_index']<2*n:
            raise RuntimeError('frozen reset requires global8 followed by previous-shot8')
        super()._install()
        torch.cuda.synchronize();started=time.perf_counter()
        active=n if self.anchor_policy=='source_only' else 2*n
        duplicate_bytes=0
        for c in primary:
            if self.anchor_policy=='source_repeat':
                c['k'][:,n:2*n].copy_(c['k'][:,:n]);c['v'][:,n:2*n].copy_(c['v'][:,:n])
                duplicate_bytes+=c['k'][:,:n].numel()*c['k'].element_size()+c['v'][:,:n].numel()*c['v'].element_size()
            c['local_end_index'].fill_(active)
            if self.anchor_policy=='source_only':
                c['pinned_start'].fill_(-1);c['pinned_len'].zero_()
            # The absolute global clock is NEVER rewound or rebased.
        torch.cuda.synchronize()
        after=cache_metadata(primary)
        expected=dict(before,local_end_index=active)
        if self.anchor_policy=='source_only':expected.update(pinned_start=-1,pinned_len=0)
        if after!=expected:raise RuntimeError('unexpected context reset metadata mutation')
        self.ledger['context_reset_wall_s']=time.perf_counter()-started
        self.ledger['context_reset_invalidated_tokens_per_head_per_layer']=before['local_end_index']-active
        self.ledger['source_replication_D2D_bytes']=duplicate_bytes
        plan=self.install['admission_plan']
        plan.update(method='privileged_native_scene_context_reset',
            active_history_before_return_tokens_per_head=active,
            preserve_original_global=self.anchor_policy=='keep',retire_previous_scene_suffix_before_denoising=True)
        if self.anchor_policy!='keep':
            multiplicity=2 if self.anchor_policy=='source_repeat' else 1
            plan.update(initial_anchor_policy=self.anchor_policy,source_multiplicity=multiplicity,
                destination_token_range=[0,active],visible_history_source_frames=plan['source_frames']*multiplicity)
        self.install.update(admission_plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            cache_metadata_unchanged=False,cache_metadata_transition=dict(before=before,after=after),
            native_attention_size_unchanged=False,native_allocated_capacity_unchanged=True,
            logical_context_changed=True,unused_suffix_excluded_by_native_endpoint_not_zero_attention=True)

    def audit(self):
        result=super().audit()
        result.update(scene_context_reset=True,logical_context_change_not_layout_optimization=True,
                      initial_anchor_policy=self.anchor_policy,
                      preserves_original_global_and_absolute_clock=self.anchor_policy=='keep',
                      preserves_absolute_clock=True,
                      observed_return_attention_shapes=[dict(query_start_latent=f,Q=q,K=k,calls=c)
                          for (f,q,k),c in sorted(self.attention_shapes.items())])
        return result
