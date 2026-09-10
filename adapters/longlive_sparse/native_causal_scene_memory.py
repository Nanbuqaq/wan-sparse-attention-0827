"""Cue-gated scene archive baseline with explicitly current-only selector inputs.

The archive producer owns raw K/V; the selector only receives SceneDescriptor.
No scene/source frame indices, future outputs, or Dense references are provided
to the selector. Source/timing choices are audited only after generation.
"""
import hashlib
import json
import time

import torch

from .native_commit_replay import owned_cpu
from .native_retimed_memory import NativeRetimedEpisodeMemory
from .native_scene_admission import SceneDescriptor,choose_scene,has_revisit_cue


class NativeCausalSceneMemory:
    def __init__(self,pipe,*,archive_budget=8*1024**3,margin=.05,minimum_cosine=.8,min_gap=32):
        if pipe.use_relative_rope or pipe.guidance_scale!=1 or pipe.quantize_kv or pipe.num_frame_per_block!=8:
            raise ValueError('qualified only for absolute-RoPE BF16 CFG1 block8')
        if pipe.local_attn_size!=32 or pipe.sink_size!=8:
            raise ValueError('qualified only for native local32/sink8')
        self.pipe=pipe;self.budget=archive_budget
        self.policy=dict(margin=margin,minimum_cosine=minimum_cosine,min_gap=min_gap)
        self.banks=[];self.last_commit=None;self.counts={};self.version=0;self.handles=[]
        self.archives=[];self.decisions=[];self.installations=[]
        self.ledger=dict(archive_D2H_KV_bytes=0,history_H2D_KV_bytes=0,condition_summary_D2H_bytes=0,
            CPU_archive_peak_tensor_bytes=0,archive_wall_s=0.,condition_summary_wall_including_readiness_s=0.,
            selector_CPU_wall_s=0.,history_install_wall_s=0.,temporal_rebind_wall_s=0.,evicted_archives=0)

    def attach(self,current_text_for_call):
        # Driver callback is invoked only with the current generator call's frame.
        # The descriptor-only choose_scene function never receives this callback.
        self.handles=[self.pipe.generator.register_forward_pre_hook(
            lambda owner,values,kwargs:self.before(owner,values,kwargs,
                current_text=current_text_for_call(int(kwargs['current_start'])//self.pipe.frame_seq_length)),with_kwargs=True),
            self.pipe.generator.register_forward_hook(self.after,with_kwargs=True)]

    def detach(self):
        for handle in self.handles:handle.remove()
        self.handles=[]

    def _prototype(self,condition):
        began=time.perf_counter();encoded=condition['prompt_embeds']
        if encoded.shape[0]!=1:raise ValueError('scene prototype requires batch1')
        tokens=encoded[0];valid=tokens.ne(0).any(-1)
        prototype=torch.nn.functional.normalize(tokens[valid].float().mean(0),dim=0).cpu()
        if not bool(torch.isfinite(prototype).all()):raise ValueError('invalid current condition prototype')
        self.ledger['condition_summary_wall_including_readiness_s']+=time.perf_counter()-began
        self.ledger['condition_summary_D2H_bytes']+=prototype.numel()*prototype.element_size()
        return prototype

    def _archive_last_scene(self,current_frame):
        last=self.last_commit
        if last is None or last['end']!=current_frame:raise RuntimeError('only just-closed committed scenes may be archived')
        began=time.perf_counter();torch.cuda.synchronize()
        caches=self.pipe.kv_cache_pos;n=8*self.pipe.frame_seq_length
        required=sum(n*c['k'].shape[0]*c['k'].shape[2]*c['k'].shape[3]*(c['k'].element_size()+c['v'].element_size()) for c in caches)
        owned_bytes=required+last['prototype'].numel()*last['prototype'].element_size()
        if owned_bytes>self.budget:raise RuntimeError('one closed scene exceeds explicit archive budget')
        while sum(b['owned_bytes'] for b in self.banks)+owned_bytes>self.budget:
            self.banks.pop(0);self.ledger['evicted_archives']+=1
        kv=[]
        for cache in caches:
            end=int(cache['local_end_index'])
            if int(cache['global_end_index'])!=current_frame*self.pipe.frame_seq_length or end<n:
                raise RuntimeError('closed-scene source bounds mismatch')
            kv.append((owned_cpu(cache['k'][:,end-n:end]),owned_cpu(cache['v'][:,end-n:end])))
        self.version+=1
        descriptor=SceneDescriptor(self.version,current_frame,last['phase'],last['prototype'])
        self.banks.append(dict(descriptor=descriptor,kv=kv,owned_bytes=owned_bytes))
        self.archives.append(dict(archive_version=self.version,source_frames=list(range(current_frame-8,current_frame)),
            source_end=current_frame,source_phase=last['phase'],KV_bytes=required,condition_prototype_bytes=owned_bytes-required))
        self.ledger['archive_D2H_KV_bytes']+=required
        self.ledger['CPU_archive_peak_tensor_bytes']=max(self.ledger['CPU_archive_peak_tensor_bytes'],sum(b['owned_bytes'] for b in self.banks))
        self.ledger['archive_wall_s']+=time.perf_counter()-began

    def before(self,owner,values,kwargs,*,current_text):
        frame=int(kwargs['current_start'])//self.pipe.frame_seq_length
        phase=self.counts.get(frame,0);self.counts[frame]=phase+1
        if phase:return
        rope_phase=float(self.pipe._dit_model.rope_temporal_offset)
        if self.last_commit is not None and rope_phase!=self.last_commit['phase']:
            self._archive_last_scene(frame)
        if any(r['current_phase']==rope_phase for r in self.installations):
            self.decisions.append(dict(at_latent=frame,selected_version=None,
                reason='revisit_already_served_this_phase_no_new_installation',scores=[]))
            return
        prototype=self._prototype(kwargs['conditional_dict']) if has_revisit_cue(current_text) else None
        descriptors=[b['descriptor'] for b in self.banks]
        began=time.perf_counter()
        decision=choose_scene(current_text,prototype,descriptors,frame,**self.policy)
        self.ledger['selector_CPU_wall_s']+=time.perf_counter()-began
        decision.update(at_latent=frame,current_text_sha256=hashlib.sha256(current_text.encode()).hexdigest())
        self.decisions.append(decision)
        if decision['selected_version'] is None:return
        # Revisit wording persists across chunks; install once per native shot
        # phase, then allow unmodified native pin/rolling evolution.
        bank=next(b for b in self.banks if b['descriptor'].archive_version==decision['selected_version'])
        source=bank['descriptor']
        install=NativeRetimedEpisodeMemory(self.pipe,mode='raw_reveal',source_end=source.source_end,
            target_start=frame,prompts=[],destination='shot')
        install.bank=bank['kv'];install.capture=dict(completed_latents=source.source_end,
            source_frames=list(range(source.source_end-8,source.source_end)),source_temporal_offset=source.source_phase,
            K_storage='already_absolute_RoPE_positioned')
        install._install()
        plan=install.install['admission_plan']
        plan.update(method='causal_cue_T5_latest_scene_history',archive_version=source.archive_version,selection_policy=self.policy)
        sha=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        install.install.update(admission_plan_sha256=sha,KV_storage_version_sha256=sha)
        self.installations.append(dict(at_latent=frame,current_phase=rope_phase,installation=install.install,transfer_ledger=install.ledger))
        self.ledger['history_H2D_KV_bytes']+=install.ledger['demand_H2D_payload_bytes']
        self.ledger['history_install_wall_s']+=install.ledger['demand_wall_s']
        self.ledger['temporal_rebind_wall_s']+=install.ledger['temporal_key_rebind_wall_s']

    def after(self,owner,values,kwargs,result):
        frame=int(kwargs['current_start'])//self.pipe.frame_seq_length
        if self.counts[frame]!=self.pipe.sampling_steps+1:return
        if not bool((kwargs['timestep']==0).all()):raise RuntimeError('scene descriptors must come from clean commits')
        self.last_commit=dict(end=frame+kwargs['noisy_image_or_video'].shape[1],
            phase=float(self.pipe._dit_model.rope_temporal_offset),prototype=self._prototype(kwargs['conditional_dict']))

    def audit(self):
        return dict(method='causal_cue_T5_latest_scene_history',archives=self.archives,decisions=self.decisions,
            installations=self.installations,ledger=self.ledger,archive_budget_bytes=self.budget,selection_policy=self.policy,
            raw_source_frames_or_target_frames_supplied_to_selector=False,
            selector_inputs='current text + current T5 mean + past descriptors only',
            future_text_or_generated_outputs_read_by_selector=False,
            native_active_KV_capacity_unchanged=True,physical_history_sparse25percent_claimed=False,
            scope='structured explicit-scene-revisit baseline; not general entity tracking or novel admission algorithm')
