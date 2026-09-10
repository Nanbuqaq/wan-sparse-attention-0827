"""Separate position-control experiment; preserve the hash-locked old controller.

Archiving/prototypes/lifetime and choose_scene are inherited or reused unchanged.
Only the selected-bank installation can preserve original K instead of rebinding.
"""
import hashlib
import json
import time

from .native_causal_scene_memory import NativeCausalSceneMemory
from .native_episode_memory import NativeEpisodeMemory
from .native_retimed_memory import NativeRetimedEpisodeMemory
from .native_scene_admission import choose_scene,has_revisit_cue


class NativeCausalPositionControl(NativeCausalSceneMemory):
    def __init__(self,pipe,*,position_policy='recent_virtual',**kwargs):
        if position_policy not in ('original','recent_virtual'):raise ValueError('unsupported position control')
        if pipe._dit_model.t_scale!=1 or pipe._dit_model.rope_method!='linear' or pipe._dit_model.original_seq_len is not None:
            raise ValueError('qualified native linear scale1 position control required')
        super().__init__(pipe,**kwargs);self.position_policy=position_policy

    def before(self,owner,values,kwargs,*,current_text):
        # Keep the frozen admission sequence verbatim; do not patch its globals.
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
        bank=next(b for b in self.banks if b['descriptor'].archive_version==decision['selected_version'])
        source=bank['descriptor']
        installer=NativeEpisodeMemory if self.position_policy=='original' else NativeRetimedEpisodeMemory
        install=installer(self.pipe,mode='raw_reveal',source_end=source.source_end,
            target_start=frame,prompts=[],destination='shot')
        install.bank=bank['kv'];install.capture=dict(completed_latents=source.source_end,
            source_frames=list(range(source.source_end-8,source.source_end)),source_temporal_offset=source.source_phase,
            K_storage='already_absolute_RoPE_positioned')
        install._install()
        plan=install.install['admission_plan']
        plan.update(method='causal_cue_T5_latest_scene_history',archive_version=source.archive_version,
                    selection_policy=self.policy,causal_position_policy=self.position_policy)
        if self.position_policy=='original':plan.update(temporal_delta=0.,source_phase=source.source_phase,current_phase=rope_phase)
        sha=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        install.install.update(admission_plan_sha256=sha,KV_storage_version_sha256=sha)
        self.installations.append(dict(at_latent=frame,current_phase=rope_phase,installation=install.install,transfer_ledger=install.ledger))
        self.ledger['history_H2D_KV_bytes']+=install.ledger['demand_H2D_payload_bytes']
        self.ledger['history_install_wall_s']+=install.ledger['demand_wall_s']
        self.ledger['temporal_rebind_wall_s']+=install.ledger.get('temporal_key_rebind_wall_s',0.)

    def audit(self):
        result=super().audit();result['position_policy']=self.position_policy
        result['position_control_is_separate_from_hash_locked_controller']=True
        return result
