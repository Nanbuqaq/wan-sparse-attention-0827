"""Declared-input prompt events for causal memory-development workloads.

Only the active segment condition reaches the generator. Known prompt strings
may be encoded before generation, with that cost charged. Cross-attention is
invalidated at events; committed history and local self-attention KV remain
unchanged. This is not the upstream interactive recache policy.
"""
from bisect import bisect_right
from contextlib import AbstractContextManager
import hashlib
import time
import torch


def validate_schedule(segments, *, block_frames=3, latent_frames=None):
    if not segments or segments[0]['start_latent'] != 0:
        raise ValueError('a schedule must start at latent0')
    starts=[s['start_latent'] for s in segments]
    if any(type(s) is not int or s < 0 or s%block_frames for s in starts) or starts != sorted(set(starts)):
        raise ValueError('distinct increasing block-aligned event positions required')
    if latent_frames is not None and starts[-1] >= latent_frames:
        raise ValueError('event lies outside generated sequence')
    if any(not isinstance(s['prompt'],str) or not s['prompt'].strip() for s in segments):
        raise ValueError('nonempty segment prompts required')
    return starts


class ScheduledPromptContext(AbstractContextManager):
    def __init__(self,pipeline,segments,*,latent_frames):
        self.pipeline=pipeline
        self.segments=[dict(s) for s in segments]
        self.starts=validate_schedule(segments,block_frames=pipeline.num_frame_per_block,latent_frames=latent_frames)
        self.encoder_wall_s=0.
        self.events=[];self.active_segment=None;self.last_start=-1
        self.conditions=[];self.encoded_unique_prompts=0

    def __enter__(self):
        self.original_encoder=self.pipeline.text_encoder.forward
        unique={}
        begin=time.perf_counter()
        for segment in self.segments:
            prompt=segment['prompt']
            if prompt not in unique:
                unique[prompt]=self.pipeline.text_encoder(text_prompts=[prompt])
            self.conditions.append(unique[prompt])
        device=next(self.pipeline.generator.parameters()).device
        if device.type=='cuda':torch.cuda.current_stream(device).synchronize()
        self.encoder_wall_s=time.perf_counter()-begin
        self.encoded_unique_prompts=len(unique)
        def cached_encoder(*args,**kwargs):
            text=kwargs.get('text_prompts',args[0] if args else None)
            if text != [self.segments[0]['prompt']]:
                raise ValueError('outer inference prompt must match the declared first segment')
            return self.conditions[0]
        self.pipeline.text_encoder.forward=cached_encoder
        def event_hook(module,args,kwargs):
            current=int(kwargs['current_start'])//self.pipeline.frame_seq_length
            if current<self.last_start:
                raise ValueError('scheduled workload does not silently support backward recache calls')
            self.last_start=current
            index=bisect_right(self.starts,current)-1
            if self.active_segment != index:
                resets=0
                if self.active_segment is not None and self.conditions[self.active_segment] is not self.conditions[index]:
                    for cache in kwargs['crossattn_cache']:
                        cache['is_init']=False
                        resets+=1
                self.events.append(dict(start_latent=current,segment=index,crossattn_cache_resets=resets,
                    prompt_sha256=hashlib.sha256(self.segments[index]['prompt'].encode()).hexdigest()))
                self.active_segment=index
            updated=dict(kwargs);updated['conditional_dict']=self.conditions[index]
            return args,updated
        self.hook=self.pipeline.generator.register_forward_pre_hook(event_hook,with_kwargs=True)
        return self

    def __exit__(self,exc_type,exc_value,tb):
        self.hook.remove();self.pipeline.text_encoder.forward=self.original_encoder
        return False

    def audit(self):
        return dict(segments=self.segments,events=self.events,preencode_wall_s=self.encoder_wall_s,
            encoded_unique_prompts=self.encoded_unique_prompts,future_generated_frames_read=False,
            only_active_text_condition_exposed=True,known_text_schedule_preencoded=True,
            self_attention_recache=False,local_and_archived_history_preserved=True,
            upstream_interactive_runtime_reproduction=False)
