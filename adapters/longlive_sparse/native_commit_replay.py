"""Offline validation of clean-commit-log KV rematerialization, not an online method."""
import gc
import time

import torch

from .history_cache import tensor_sha256
from .offline_eval import output_error_metrics


def owned_cpu(tensor):
    return tensor.detach().to('cpu',copy=True).contiguous()


def cache_samples(caches):
    rows=[]
    for layer in (0,9,19,29):
        cache=caches[layer];end=int(cache['local_end_index'])
        index=torch.linspace(0,end-1,min(256,end),device=cache['k'].device).round().long()
        rows.append(dict(layer=layer,K=owned_cpu(cache['k'][0,:,0].index_select(0,index)),
            V=owned_cpu(cache['v'][0,:,0].index_select(0,index))))
    return rows


def cache_metadata(caches):
    return {k:int(caches[0][k]) for k in ('global_end_index','local_end_index','pinned_start','pinned_len')}


class NativeCleanCommitLog:
    def __init__(self,pipeline):
        self.pipeline=pipeline;self.records=[];self.conditions={};self.capture_bytes=0

    def hook(self,owner,values,kwargs,result):
        timestep=kwargs['timestep']
        if not bool((timestep==0).all()):return
        latent=owned_cpu(kwargs['noisy_image_or_video'])
        condition=owned_cpu(kwargs['conditional_dict']['prompt_embeds']);key=tensor_sha256(condition)
        self.conditions.setdefault(key,condition)
        model=self.pipeline._dit_model
        settings={k:getattr(model,k) for k in ('local_attn_size','t_scale','rope_method','original_seq_len','use_relative_rope','rope_temporal_offset')}
        samples=cache_samples(self.pipeline.kv_cache_pos)
        self.capture_bytes+=latent.numel()*latent.element_size()+condition.numel()*condition.element_size()
        self.capture_bytes+=sum(x[k].numel()*x[k].element_size() for x in samples for k in ('K','V'))
        self.records.append(dict(latent=latent,timestep=owned_cpu(timestep),condition_key=key,
            current_start=int(kwargs['current_start']),cache_start=int(kwargs.get('cache_start',kwargs['current_start'])),
            settings=settings,samples=samples,metadata=cache_metadata(self.pipeline.kv_cache_pos)))

    @torch.inference_mode()
    def replay_and_compare(self,*,prompts,returned_latent):
        pipe=self.pipeline;device=returned_latent.device;dtype=returned_latent.dtype
        if len(self.records)!=returned_latent.shape[1]//pipe.num_frame_per_block:
            raise ValueError('exactly one captured clean commit per chunk required')
        if any(c.get('quantized',False) for c in pipe.kv_cache_pos):raise ValueError('BF16 cache replay only')
        # Full witness is diagnostic only. It is not part of the proposed log.
        witness=[{k:owned_cpu(v) for k,v in c.items() if k in ('k','v','global_end_index','local_end_index','pinned_start','pinned_len')} for c in pipe.kv_cache_pos]
        witness_bytes=sum(c[k].numel()*c[k].element_size() for c in witness for k in ('k','v'))
        pipe.kv_cache_pos=pipe.kv_cache_neg=pipe.crossattn_cache_pos=pipe.crossattn_cache_neg=None
        gc.collect();torch.cuda.empty_cache()
        pipe._set_all_modules_max_attention_size(pipe.local_attn_size)
        pipe._set_all_modules_sink_size(pipe.sink_size)
        pipe._set_all_modules_global_sink_size(pipe.global_sink_size)
        torch.cuda.synchronize();started=time.perf_counter()
        pipe._initialize_kv_cache(batch_size=1,dtype=dtype,device=device)
        pipe._initialize_crossattn_cache(batch_size=1,dtype=dtype,device=device)
        torch.cuda.synchronize();allocation_s=time.perf_counter()-started
        rows=[];replay_service=0.;h2d_bytes=0
        for i,record in enumerate(self.records):
            for k,v in record['settings'].items():setattr(pipe._dit_model,k,v)
            for caches in (pipe.crossattn_cache_pos,pipe.crossattn_cache_neg):
                for c in caches:c['is_init']=False
            torch.cuda.synchronize();begin=time.perf_counter()
            x=record['latent'].to(device);t=record['timestep'].to(device)
            condition=self.conditions[record['condition_key']].to(device)
            h2d_bytes+=sum(z.numel()*z.element_size() for z in (x,t,condition))
            pipe.generator(noisy_image_or_video=x,conditional_dict={'prompt_embeds':condition},timestep=t,
                kv_cache=pipe.kv_cache_pos,crossattn_cache=pipe.crossattn_cache_pos,
                current_start=record['current_start'],cache_start=record['cache_start'])
            torch.cuda.synchronize();service=time.perf_counter()-begin;replay_service+=service
            actual=cache_samples(pipe.kv_cache_pos);comparisons=[]
            for left,right in zip(record['samples'],actual):
                comparisons.append(dict(layer=left['layer'],K_exact=torch.equal(left['K'],right['K']),
                    V_exact=torch.equal(left['V'],right['V']),K_error=output_error_metrics(left['K'],right['K']),
                    V_error=output_error_metrics(left['V'],right['V'])))
            frame=record['current_start']//pipe.frame_seq_length
            represented=returned_latent[:,frame:frame+x.shape[1]].detach().cpu()
            rows.append(dict(chunk=i,start_latent=frame,samples=comparisons,
                metadata_before_pin_exact=record['metadata']==cache_metadata(pipe.kv_cache_pos),
                original_clean_input_dtype=str(record['latent'].dtype),returned_latent_dtype=str(represented.dtype),
                clean_input_matches_returned_latent=torch.equal(record['latent'],represented),
                upload_and_forward_service_s=service))
            if pipe._is_scene_cut(prompts,i):pipe._pin_current_chunk(pipe.kv_cache_pos,x.shape[1])
        full=[]
        for layer,(before,after) in enumerate(zip(witness,pipe.kv_cache_pos)):
            key,value=owned_cpu(after['k']),owned_cpu(after['v'])
            full.append(dict(layer=layer,K_exact=torch.equal(before['k'],key),V_exact=torch.equal(before['v'],value),
                K_error=output_error_metrics(before['k'],key),V_error=output_error_metrics(before['v'],value),
                metadata_exact=all(torch.equal(before[k],after[k].cpu()) for k in ('global_end_index','local_end_index','pinned_start','pinned_len'))))
        exact=all(r['K_exact'] and r['V_exact'] and r['metadata_exact'] for r in full)
        log_bytes=sum(r['latent'].numel()*r['latent'].element_size()+r['timestep'].numel()*r['timestep'].element_size() for r in self.records)
        log_bytes+=sum(t.numel()*t.element_size() for t in self.conditions.values())
        return dict(status='pass' if exact else 'negative',full_final_cache_bitwise_exact=exact,
            clean_commit_count=len(self.records),sampled_step_comparisons=rows,full_final_cache_comparisons=full,
            log_tensor_bytes=log_bytes,log_metadata_bytes_not_included=True,diagnostic_witness_KV_bytes=witness_bytes,
            witness_KV_over_log_tensor_bytes=witness_bytes/log_bytes,observer_D2H_bytes=self.capture_bytes+witness_bytes,
            replay_H2D_bytes=h2d_bytes,cache_allocation_s=allocation_s,replay_upload_and_forward_service_s=replay_service,
            complete_audit_wall_s=time.perf_counter()-started,full_reference_and_sample_comparisons_excluded_from_service=True,
            scope='offline_validation_not_deployed_memory_or_video_speedup',full_KV_witness_is_not_a_log_requirement=True,
            numerical_or_context_mismatch_must_not_be_hidden=True)
