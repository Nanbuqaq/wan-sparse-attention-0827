"""Privileged past-episode frame intervention for diagnosing retrieval lag.

No future frames or Dense outputs are read, but the anchor interval is supplied
by the development workload. This is a diagnostic teacher, not an automatic
online memory method or an eligible speed-Pareto candidate.
"""
from contextlib import AbstractContextManager
import torch


class EpisodeAnchorProbe(AbstractContextManager):
    def __init__(self,pipeline,*,anchor_frames,start_latent,end_latent):
        if not anchor_frames or len(set(anchor_frames))!=len(anchor_frames) or min(anchor_frames)<0:
            raise ValueError('distinct nonnegative past anchor frames required')
        if max(anchor_frames)>=start_latent or end_latent<=start_latent:
            raise ValueError('anchors must precede a nonempty intervention interval')
        self.pipeline=pipeline;self.anchors=tuple(sorted(anchor_frames))
        self.start=start_latent;self.end=end_latent;self.records={}

    def __enter__(self):
        def hook(module,args,kwargs):
            frame=int(kwargs['current_start'])//self.pipeline.frame_seq_length
            if not self.start<=frame<self.end:return None
            archive=self.pipeline.sparse_history_archive
            for owner in self.pipeline.sparse_history_modules:
                if not set(self.anchors)<=set(archive.frame_ids(owner.layer_id)):
                    raise RuntimeError('anchor is not in every layer of the already committed archive')
            sink=int(self.pipeline.sparse_history_modules[0].sink_size)
            memory_size=int(self.pipeline.sparse_history_modules[0].memory_size)
            if len(self.anchors)!=memory_size:
                raise ValueError('diagnostic must preserve the configured coarse-frame count')
            prior=kwargs.get('memory_indices')
            original=None if prior is None else (prior.detach().cpu()+sink).tolist()
            device=kwargs['noisy_image_or_video'].device
            batch=kwargs['noisy_image_or_video'].shape[0]
            indices=torch.tensor([f-sink for f in self.anchors],device=device,dtype=torch.long)[None].expand(batch,-1)
            if min(self.anchors)<sink:raise ValueError('sink frames cannot be addressed as archive slots')
            if frame not in self.records:
                self.records[frame]=dict(query_latent=frame,original_proposed_frames=original,
                    effective_anchor_frames=list(self.anchors),calls=0)
            self.records[frame]['calls']+=1
            changed=dict(kwargs);changed['memory_indices']=indices
            return args,changed
        self.hook=self.pipeline.generator.register_forward_pre_hook(hook,with_kwargs=True)
        return self

    def __exit__(self,exc_type,exc_value,tb):
        self.hook.remove();return False

    def audit(self):
        return dict(anchor_frames=list(self.anchors),start_latent=self.start,end_latent=self.end,
            records=[self.records[k] for k in sorted(self.records)],future_frame_access=False,
            full_Dense_output_access=False,privileged_episode_interval=True,automatic_online_method=False,
            original_pipeline_retrieval_log_contains_proposals_not_overrides=True)
