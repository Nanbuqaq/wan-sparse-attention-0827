"""Inference adapter for the stage1 A/B/C native five-latent checkpoints.

The caller owns source/generation clocks and the verified buffered VAE. A
block has the same fields as h3_stage1_abc_data.Reader.sample entries. Video
and audio values are SHAPE templates only: generation starts from noise.
"""
import torch
from h3_fixed_kv_stage1 import Context,VideoBank,forward
from h3_stage1_abc_data import pack


class FixedKVStream:
    def __init__(self,transformer,strategy='A',history_seconds=8.):
        if strategy not in ('A','B','C') or history_seconds!=8.:
            raise ValueError('Use the checkpoint-declared strategy and eight-second history')
        self.transformer=transformer;self.strategy=strategy;self.history_seconds=history_seconds
        self.bank=VideoBank(detach=True);self.blocks={};self.last_block=-1

    def reset(self):
        self.bank=VideoBank(detach=True);self.blocks={};self.last_block=-1

    @torch.no_grad()
    def generate(self,block,generator,device,nfe=50):
        if nfe!=50:raise ValueError('This release requires native 50-NFE sampling, CFG1')
        block=dict(block);i=block['block']
        if i<=self.last_block:raise ValueError('Current block must follow all previously committed blocks')
        past=tuple(j for j,b in self.blocks.items() if b['start']>=block['source_start']-8.-1e-7
                   and b['end']<=block['source_start']+1e-7)
        self.bank.evict(past)
        block.update(past=past,target=True)
        batch=pack([block],device,generator,strategy=self.strategy,include_clean=False)
        inputs=batch['inputs'];video=torch.randn(inputs['hidden_states'].shape,device=device,generator=generator)
        audio=torch.randn(inputs['audio_hidden_states'].shape,device=device,generator=generator)
        context=Context(batch['context'].splits,self.strategy=='B',self.bank.snapshot())
        for step in range(nfe):
            u=1-step/nfe;un=1-(step+1)/nfe
            tv,ta=1-12*u/(1+11*u),1-3*u/(1+2*u)
            nv,na=1-12*un/(1+11*un),1-3*un/(1+2*un)
            clock=torch.ones(len(inputs['position_ids']),device=device)
            clock[inputs['video_indices']]=tv;clock[inputs['audio_indices']]=ta
            inputs['timestep'],inputs['timestep_indices']=torch.unique(clock,sorted=True,return_inverse=True)
            inputs['hidden_states'],inputs['audio_hidden_states']=video,audio
            pv,pa=forward(self.transformer,inputs,context)
            video=video+(nv-tv)*pv.float();audio=audio+(na-ta)*pa.float()
        if not bool(torch.isfinite(video).all()) or not bool(torch.isfinite(audio).all()):
            raise RuntimeError('Nonfinite generated chunk')
        # Commit clean VIDEO only; do not allow clean/future audio into history.
        clean_block=dict(block,video=video[0].float())
        clean=pack([clean_block],device,generator,strategy=self.strategy,clean_only=True)
        context=Context(clean['context'].splits,self.strategy=='B',self.bank.snapshot(),self.bank)
        forward(self.transformer,clean['inputs'],context)
        self.blocks[i]={'start':block['source_start'],'end':block['source_end']};self.last_block=i
        # Keep no latent history and no codec buffer in this class.
        return video,audio,dict(block=i,read_past_blocks=past,KV_bytes=self.bank.bytes,
            clean_commit_forwards=1,audio_KV_stored=False,old_video_queries_recomputed=False,
            clean_time_video=1.,codec_reset_requested=False)
