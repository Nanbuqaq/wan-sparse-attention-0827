"""Privileged native5B episode admission with raw or clean-log physical storage.

The admission is a diagnostic, not an autonomous selector. Stored K is already
absolute-RoPE positioned; physical relocation MUST NOT apply RoPE a second time.
"""
import hashlib
import json
import time

import torch

from .history_cache import tensor_sha256
from .native_commit_replay import owned_cpu,cache_metadata


def admission_plan(*,source_end,target_start,frames,frame_tokens,layers,heads,dim,dtype):
    plan=dict(method='privileged_native_episode_prefix',source_frames=list(range(source_end-frames,source_end)),
        target_start=target_start,destination_token_range=[0,frames*frame_tokens],layers=layers,
        heads=heads,head_dim=dim,dtype=str(dtype),position_policy='preserve_stored_absolute_RoPE',
        frame_tokens=frame_tokens,attention_capacity_unchanged=True)
    sha=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return plan,sha


class NativeEpisodeMemory:
    def __init__(self,pipe,*,mode,source_end,target_start,prompts,raw_budget=3*1024**3,log_budget=64*1024**2):
        if mode not in ('none','raw_reveal','raw_away','log_reveal'):raise ValueError('unknown episode mode')
        if pipe.use_relative_rope or pipe.guidance_scale!=1 or pipe.quantize_kv or pipe.num_frame_per_block!=8:
            raise ValueError('episode probe requires native BF16 absolute-RoPE CFG1 block8')
        self.pipe,self.mode=pipe,mode;self.frames=8;self.target=target_start
        # The first non-pinned away block is no longer resident at return. This
        # avoids a wrong-memory control that merely duplicates an existing KV.
        if target_start-source_end<32 or pipe.local_attn_size!=32 or pipe.sink_size!=8:
            raise ValueError('the frozen unique-away control needs local32/sink8 and >=32-away latents')
        self.source_end=source_end+16 if mode=='raw_away' else source_end
        if not self.source_end<=self.target or self.source_end<8:raise ValueError('uncommitted episode bounds')
        self.prompts=prompts;self.raw_budget=raw_budget;self.log_budget=log_budget
        self.counts={};self.busy=False;self.bank=None;self.records=[];self.conditions={}
        self.capture=None;self.install=None;self.handles=[]
        self.ledger=dict(archive_D2H_payload_bytes=0,demand_H2D_payload_bytes=0,demand_D2D_KV_bytes=0,
            archive_wall_s=0.,demand_wall_s=0.,CPU_archive_peak_bytes=0,
            native_control_metadata_traffic_not_in_payload_ledger=True)

    def attach(self):
        self.handles=[self.pipe.generator.register_forward_pre_hook(self.before,with_kwargs=True),
            self.pipe.generator.register_forward_hook(self.after,with_kwargs=True)]

    def detach(self):
        for handle in self.handles:handle.remove()
        self.handles=[]

    def before(self,owner,values,kwargs):
        if self.busy:return
        start=int(kwargs['current_start']);phase=self.counts.get(start,0);self.counts[start]=phase+1
        if self.mode!='none' and start==self.target*self.pipe.frame_seq_length and phase==0:
            if self.install is not None:raise RuntimeError('duplicate episode installation')
            self._install()

    def after(self,owner,values,kwargs,result):
        if self.busy or self.mode=='none':return
        start=int(kwargs['current_start'])
        if self.counts[start]!=self.pipe.sampling_steps+1:return
        if not bool((kwargs['timestep']==0).all()):raise RuntimeError('episode archive must use clean commits')
        end=start//self.pipe.frame_seq_length+kwargs['noisy_image_or_video'].shape[1]
        if end>self.source_end:return
        began=time.perf_counter()
        if self.mode=='log_reveal':
            x=owned_cpu(kwargs['noisy_image_or_video']);t=owned_cpu(kwargs['timestep'])
            c=owned_cpu(kwargs['conditional_dict']['prompt_embeds']);key=tensor_sha256(c)
            self.conditions.setdefault(key,c)
            settings={k:getattr(self.pipe._dit_model,k) for k in ('local_attn_size','t_scale','rope_method','original_seq_len','use_relative_rope','rope_temporal_offset')}
            chunk=start//self.pipe.frame_seq_length//8
            self.records.append(dict(latent=x,timestep=t,condition_key=key,current_start=start,
                cache_start=int(kwargs.get('cache_start',start)),settings=settings,
                pin_after=self.pipe._is_scene_cut(self.prompts[:chunk+1],chunk)))
            self.ledger['archive_D2H_payload_bytes']+=sum(z.numel()*z.element_size() for z in (x,t,c))
            size=sum(r['latent'].numel()*r['latent'].element_size()+r['timestep'].numel()*r['timestep'].element_size() for r in self.records)
            size+=sum(v.numel()*v.element_size() for v in self.conditions.values())
            if size>self.log_budget:raise RuntimeError('episode log exceeds explicit budget')
            self.ledger['CPU_archive_peak_bytes']=max(size,self.ledger['CPU_archive_peak_bytes'])
        if end==self.source_end:
            caches=self.pipe.kv_cache_pos;n=self.frames*self.pipe.frame_seq_length
            if int(caches[0]['global_end_index'])!=end*self.pipe.frame_seq_length:raise RuntimeError('source coordinates mismatch')
            if self.mode.startswith('raw_'):
                required=sum(n*c['k'].shape[0]*c['k'].shape[2]*c['k'].shape[3]*(c['k'].element_size()+c['v'].element_size()) for c in caches)
                if required>self.raw_budget:raise RuntimeError('episode raw snapshot exceeds explicit budget')
                self.bank=[(owned_cpu(c['k'][:,int(c['local_end_index'])-n:int(c['local_end_index'])]),
                    owned_cpu(c['v'][:,int(c['local_end_index'])-n:int(c['local_end_index'])])) for c in caches]
                self.ledger['archive_D2H_payload_bytes']+=required;self.ledger['CPU_archive_peak_bytes']=required
            self.capture=dict(completed_latents=end,source_frames=list(range(end-self.frames,end)),
                K_storage='already_absolute_RoPE_positioned',source_temporal_offset=float(self.pipe._dit_model.rope_temporal_offset))
        self.ledger['archive_wall_s']+=time.perf_counter()-began

    def _install(self):
        if self.capture is None:raise RuntimeError('episode is not committed')
        pipe=self.pipe;primary=pipe.kv_cache_pos;before=cache_metadata(primary)
        n=self.frames*pipe.frame_seq_length
        if n!=pipe.global_sink_size*pipe.frame_seq_length:raise RuntimeError('replace exactly the existing global prefix')
        if any(c.get('quantized',False) for c in primary):raise RuntimeError('quantized cache unsupported')
        plan,sha=admission_plan(source_end=self.source_end,target_start=self.target,frames=self.frames,
            frame_tokens=pipe.frame_seq_length,layers=len(primary),heads=primary[0]['k'].shape[2],
            dim=primary[0]['k'].shape[3],dtype=primary[0]['k'].dtype)
        torch.cuda.synchronize();began=time.perf_counter();self.busy=True
        saved_caches={k:getattr(pipe,k) for k in ('kv_cache_pos','kv_cache_neg','crossattn_cache_pos','crossattn_cache_neg')}
        saved_settings={k:getattr(pipe._dit_model,k) for k in ('local_attn_size','t_scale','rope_method','original_seq_len','use_relative_rope','rope_temporal_offset')}
        try:
            if self.mode.startswith('raw_'):
                for c,(key,value) in zip(primary,self.bank):
                    c['k'][:,:n].copy_(key);c['v'][:,:n].copy_(value)
                    self.ledger['demand_H2D_payload_bytes']+=key.numel()*key.element_size()+value.numel()*value.element_size()
            else:
                with torch.random.fork_rng(devices=[primary[0]['k'].device]):
                    pipe._initialize_kv_cache(batch_size=1,dtype=primary[0]['k'].dtype,device=primary[0]['k'].device)
                    pipe._initialize_crossattn_cache(batch_size=1,dtype=primary[0]['k'].dtype,device=primary[0]['k'].device)
                    for record in self.records:
                        if record['current_start']//pipe.frame_seq_length+record['latent'].shape[1]>self.source_end:
                            raise RuntimeError('replay read beyond the closed episode')
                        for k,v in record['settings'].items():setattr(pipe._dit_model,k,v)
                        for caches in (pipe.crossattn_cache_pos,pipe.crossattn_cache_neg):
                            for c in caches:c['is_init']=False
                        x=record['latent'].to(primary[0]['k'].device);t=record['timestep'].to(x.device)
                        condition=self.conditions[record['condition_key']].to(x.device)
                        self.ledger['demand_H2D_payload_bytes']+=sum(z.numel()*z.element_size() for z in (x,t,condition))
                        pipe.generator(noisy_image_or_video=x,conditional_dict={'prompt_embeds':condition},timestep=t,
                            kv_cache=pipe.kv_cache_pos,crossattn_cache=pipe.crossattn_cache_pos,
                            current_start=record['current_start'],cache_start=record['cache_start'])
                        if record['pin_after']:
                            type(pipe)._pin_current_chunk(pipe,pipe.kv_cache_pos,x.shape[1])
                    for c,scratch in zip(primary,pipe.kv_cache_pos):
                        end=int(scratch['local_end_index'])
                        c['k'][:,:n].copy_(scratch['k'][:,end-n:end]);c['v'][:,:n].copy_(scratch['v'][:,end-n:end])
                        self.ledger['demand_D2D_KV_bytes']+=2*c['k'][:,:n].numel()*c['k'].element_size()
            torch.cuda.synchronize()
        finally:
            for k,v in saved_caches.items():setattr(pipe,k,v)
            for k,v in saved_settings.items():setattr(pipe._dit_model,k,v)
            self.busy=False
        self.ledger['demand_wall_s']=time.perf_counter()-began
        if cache_metadata(primary)!=before:raise RuntimeError('episode changed cache positions/endpoints')
        self.install=dict(at_latent=self.target,admission_plan=plan,admission_plan_sha256=sha,
            transfer_mode=self.mode,cache_metadata_unchanged=True,installed_once=True,
            native_attention_size_unchanged=True)

    def audit(self):
        if self.mode!='none' and self.install is None:raise RuntimeError('missing episode demand event')
        return dict(mode=self.mode,privileged_admission_not_autonomous=True,capture=self.capture,installation=self.install,
            ledger=self.ledger,raw_budget_bytes=self.raw_budget,log_budget_bytes=self.log_budget,
            K_positions_preserved_not_rebased=True,no_future_generated_data_used=True)
