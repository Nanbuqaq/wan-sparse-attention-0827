"""Causal isolated-latent re-encoding, not exact replay of original history KV.

The source-only policy invalidates all old context before admission, allowing
its existing GPU buffers to serve as temporary re-encoding workspace. No
second full KV cache is allocated. Stored latent is from a completed chunk.
"""
import hashlib
import json
import time

import torch

from .history_cache import tensor_sha256
from .native_commit_replay import owned_cpu,cache_metadata
from .native_scene_context import NativeSceneContextReset


class NativeSemanticRematMemory(NativeSceneContextReset):
    def __init__(self,pipe,*,reconstruction_condition,CPU_budget=16*1024**2,**kwargs):
        if reconstruction_condition not in ('past','current'):raise ValueError('unknown reconstruction condition')
        if kwargs.get('anchor_policy')!='source_repeat_pinned':raise ValueError('this diagnostic uses the pinned repeated-source route')
        super().__init__(pipe,**kwargs)
        self.reconstruction_condition=reconstruction_condition;self.CPU_budget=CPU_budget
        self.source_latent=None;self.source_condition=None;self.source_settings=None
        self.ledger.update(reconstruction_H2D_bytes=0,reconstruction_condition_hash_D2H_bytes=0,
                           reconstruction_wall_s=0.,reconstruction_calls=0)

    def after(self,owner,values,kwargs,result):
        if self.busy or self.source_latent is not None:return
        start=int(kwargs['current_start'])
        if self.counts[start]!=self.pipe.sampling_steps+1:return
        end=start//self.pipe.frame_seq_length+kwargs['noisy_image_or_video'].shape[1]
        if end!=self.source_end:return
        if not bool((kwargs['timestep']==0).all()):raise RuntimeError('source must be a completed clean latent')
        x=kwargs['noisy_image_or_video'];c=kwargs['conditional_dict']['prompt_embeds']
        needed=x.numel()*x.element_size()+c.numel()*c.element_size()
        if needed>self.CPU_budget:raise RuntimeError('source latent/condition exceeds explicit CPU budget')
        started=time.perf_counter();self.source_latent=owned_cpu(x);self.source_condition=owned_cpu(c)
        self.source_settings={k:getattr(self.pipe._dit_model,k) for k in ('local_attn_size','t_scale','rope_method','original_seq_len','use_relative_rope','rope_temporal_offset')}
        self.ledger['archive_wall_s']=time.perf_counter()-started
        self.ledger['archive_D2H_payload_bytes']=needed;self.ledger['CPU_archive_peak_bytes']=needed
        self.capture=dict(completed_latents=end,source_frames=list(range(end-8,end)),
            representation='actual_clean_latent_and_past_condition_not_KV',
            latent_sha256=tensor_sha256(self.source_latent),condition_sha256=tensor_sha256(self.source_condition),
            source_temporal_offset=self.source_settings['rope_temporal_offset'])

    def before(self,owner,values,kwargs):
        if self.busy:return
        self.current_condition=kwargs['conditional_dict']['prompt_embeds']
        super().before(owner,values,kwargs)

    def _install(self):
        if self.source_latent is None:raise RuntimeError('source latent not committed')
        pipe=self.pipe;primary=pipe.kv_cache_pos;before=cache_metadata(primary);n=8*pipe.frame_seq_length
        pointers=[(c['k'].data_ptr(),c['v'].data_ptr()) for c in primary]
        settings={k:getattr(pipe._dit_model,k) for k in self.source_settings}
        source_start=self.source_end-8;self.busy=True;torch.cuda.synchronize();started=time.perf_counter()
        try:
            # Old context has no logical role after this policy's transition.
            # Mutate only metadata here; native forward overwrites active data.
            for cache in primary:
                cache['global_end_index'].fill_(source_start*pipe.frame_seq_length)
                cache['local_end_index'].zero_();cache['pinned_start'].fill_(-1);cache['pinned_len'].zero_()
            for cache in pipe.crossattn_cache_pos:cache['is_init']=False
            for k,v in self.source_settings.items():setattr(pipe._dit_model,k,v)
            device=primary[0]['k'].device
            x=self.source_latent.to(device)
            condition=self.source_condition.to(device) if self.reconstruction_condition=='past' else self.current_condition
            used_condition=owned_cpu(condition)
            condition_hash=tensor_sha256(used_condition)
            self.ledger['reconstruction_condition_hash_D2H_bytes']=condition.numel()*condition.element_size()
            self.ledger['reconstruction_H2D_bytes']=x.numel()*x.element_size()
            if self.reconstruction_condition=='past':self.ledger['reconstruction_H2D_bytes']+=condition.numel()*condition.element_size()
            self.ledger['CPU_archive_peak_bytes']+=used_condition.numel()*used_condition.element_size()
            if self.ledger['CPU_archive_peak_bytes']>self.CPU_budget:raise RuntimeError('temporary condition witness exceeds budget')
            with torch.random.fork_rng(devices=[device]):
                pipe.generator(noisy_image_or_video=x,conditional_dict={'prompt_embeds':condition},
                    timestep=torch.zeros((x.shape[0],x.shape[1]),device=device,dtype=torch.float32),
                    kv_cache=primary,crossattn_cache=pipe.crossattn_cache_pos,
                    current_start=source_start*pipe.frame_seq_length,cache_start=source_start*pipe.frame_seq_length)
            self.ledger['reconstruction_calls']=1
            if cache_metadata(primary)['global_end_index']!=self.source_end*pipe.frame_seq_length or cache_metadata(primary)['local_end_index']!=n:
                raise RuntimeError('isolated source did not produce exactly eight KV frames')
            for cache in primary:
                cache['k'][:,n:2*n].copy_(cache['k'][:,:n]);cache['v'][:,n:2*n].copy_(cache['v'][:,:n])
                cache['global_end_index'].fill_(self.target*pipe.frame_seq_length)
                cache['local_end_index'].fill_(2*n);cache['pinned_start'].fill_(n);cache['pinned_len'].fill_(n)
            torch.cuda.synchronize()
        finally:
            for k,v in settings.items():setattr(pipe._dit_model,k,v)
            for cache in pipe.crossattn_cache_pos:cache['is_init']=False
            self.busy=False
        if pointers!=[(c['k'].data_ptr(),c['v'].data_ptr()) for c in primary]:raise RuntimeError('allocated a second KV cache instead of reusing storage')
        self.ledger['reconstruction_wall_s']=time.perf_counter()-started
        self.ledger['demand_wall_s']=self.ledger['reconstruction_wall_s']
        self.ledger['demand_H2D_payload_bytes']=self.ledger['reconstruction_H2D_bytes']
        self.ledger['source_replication_D2D_bytes']=sum(2*c['k'][:,:n].numel()*c['k'].element_size() for c in primary)
        plan=dict(method='privileged_isolated_latent_reencode',source_frames=self.capture['source_frames'],
            source_multiplicity=2,initial_anchor_policy='source_repeat_pinned',reconstruction_condition=self.reconstruction_condition,
            used_condition_sha256=condition_hash,source_latent_sha256=self.capture['latent_sha256'],
            source_context='empty_before_source_clean_forward',original_KV_preserved=False,
            position_policy='source_absolute_RoPE_preserved_current_clock_restored',source_settings=self.source_settings,
            destination_token_range=[0,2*n],target_start=self.target)
        sha=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.install=dict(at_latent=self.target,admission_plan=plan,admission_plan_sha256=sha,
            KV_storage_version_sha256=sha,cache_metadata_transition=dict(before=before,after=cache_metadata(primary)),
            cache_metadata_unchanged=False,native_allocated_capacity_unchanged=True,all_GPU_KV_storage_pointers_preserved=True,
            not_exact_original_KV_replay=True)

    def audit(self):
        result=super().audit()
        result.update(memory_representation='isolated_latent_reencode',reconstruction_condition=self.reconstruction_condition,
            reconstruction_CPU_budget_bytes=self.CPU_budget,original_KV_values_preserved=False,
            no_future_data=True,current_condition_known_before_current_forward=True)
        return result
