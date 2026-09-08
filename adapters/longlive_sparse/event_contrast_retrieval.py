"""Causal event-conditioned retrieval from past instruction episodes.

This experimental selector receives only the current condition and committed
archive metadata. It has no workload-role labels or predefined anchor interval.
It changes retrieval only for the first chunk of a new condition. At least two
closed instruction episodes are required; all other calls use ordinary RAG.
"""
from contextlib import AbstractContextManager
import hashlib
import time
import torch
import torch.nn.functional as F


class EventContrastRetrieval(AbstractContextManager):
    def __init__(self,pipeline,*,policy='contrast'):
        if policy not in ('contrast','cosine'):raise ValueError('unsupported event-key policy')
        self.policy=policy
        self.pipeline=pipeline;self.condition=None;self.episodes=[];self.events=[]
        self.pending=None;self.last_frame=-1;self.wall_s=0.;self.metadata_D2H_bytes=0;self.metadata_H2D_bytes=0

    def __enter__(self):
        def hook(module,args,kwargs):
            started=time.perf_counter();frame=int(kwargs['current_start'])//self.pipeline.frame_seq_length
            if frame<self.last_frame:raise ValueError('event contrast does not silently support backward recache')
            self.last_frame=frame
            condition=kwargs['conditional_dict']
            if condition is not self.condition:
                if self.episodes:self.episodes[-1]['end']=frame
                embedding=condition['prompt_embeds'].detach().float()
                if embedding.shape[0]!=1:raise ValueError('event-contrast development currently supports one stream')
                valid=embedding.abs().sum(-1)>0
                vector=(embedding*valid[...,None]).sum(1)/valid.sum(1).clamp_min(1)[...,None]
                key=F.normalize(vector,dim=-1)[0].cpu()
                self.metadata_D2H_bytes+=key.numel()*key.element_size()
                prior=list(self.episodes)
                self.episodes.append(dict(start=frame,end=None,key=key))
                self.condition=condition;self.pending=None
                event=dict(start_latent=frame,prior_episode_count=len(prior),mode='ordinary_RAG',
                    condition_key_sha256=hashlib.sha256(key.numpy().tobytes()).hexdigest())
                if len(prior)>=2:
                    keys=torch.stack([e['key'] for e in prior])
                    cosine=keys@key;contrast=cosine-keys@prior[-1]['key']
                    chosen=int((contrast if self.policy=='contrast' else cosine).argmax());episode=prior[chosen]
                    archive=self.pipeline.sparse_history_archive
                    shared=set(archive.frame_ids(self.pipeline.sparse_history_modules[0].layer_id))
                    for owner in self.pipeline.sparse_history_modules[1:]:shared.intersection_update(archive.frame_ids(owner.layer_id))
                    available=sorted(f for f in shared if episode['start']<=f<episode['end'] and f<frame)
                    count=int(self.pipeline.sparse_history_modules[0].memory_size)
                    event.update(cosine_scores=cosine.tolist(),contrast_scores=contrast.tolist(),chosen_episode=chosen,
                        chosen_episode_start=episode['start'],chosen_episode_end=episode['end'])
                    if count>0 and len(available)>=count:
                        anchors=available[-count:]
                        self.pending=(frame,frame+self.pipeline.num_frame_per_block,anchors)
                        event.update(mode='event_'+self.policy,effective_frames=anchors)
                    else:event['fallback_reason']='insufficient_committed_frames_in_chosen_episode'
                self.events.append(event)
            changed=None
            if self.pending is not None and self.pending[0]<=frame<self.pending[1]:
                anchors=self.pending[2];sink=int(self.pipeline.sparse_history_modules[0].sink_size)
                device=kwargs['noisy_image_or_video'].device
                indices=torch.tensor([[f-sink for f in anchors]],device=device,dtype=torch.long)
                self.metadata_H2D_bytes+=indices.numel()*indices.element_size()
                changed=dict(kwargs);changed['memory_indices']=indices
            self.wall_s+=time.perf_counter()-started
            return (args,changed) if changed is not None else None
        self.hook=self.pipeline.generator.register_forward_pre_hook(hook,with_kwargs=True)
        return self

    def __exit__(self,exc_type,exc_value,tb):self.hook.remove();return False

    def audit(self):
        overrides=[e['start_latent'] for e in self.events if e['mode'] in ('event_contrast','event_cosine')]
        return dict(policy=self.policy,events=self.events,
            selector_wall_s=self.wall_s,condition_key_D2H_bytes=self.metadata_D2H_bytes,index_H2D_bytes=self.metadata_H2D_bytes,
            first_override_latent=min(overrides) if overrides else None,
            past_episode_keys_only=True,current_condition_only=True,workload_role_labels_used=False,
            predefined_anchor_interval_used=False,future_video_used=False,automatic_experimental_method=True,
            episode_table_bounded=False,ordinary_visual_RAG_between_events=True)
