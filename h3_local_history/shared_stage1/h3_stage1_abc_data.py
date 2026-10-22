"""Native H3 five-latent data packs, independent text states and matched flow."""
from collections import OrderedDict
from pathlib import Path
import math

import torch

from h3_fixed_kv_stage1 import Context,Split


class Reader:
    def __init__(self, limit_bytes=4*2**30):
        self.cache=OrderedDict();self.limit=limit_bytes;self.bytes=0

    def tensor(self,path):
        if path not in self.cache:
            value=torch.load(path,map_location='cpu',weights_only=True)
            size=sum(v.numel()*v.element_size() for v in value.values() if isinstance(v,torch.Tensor)) if isinstance(value,dict) else value.numel()*value.element_size()
            self.cache[path]=(value,size);self.bytes+=size
            while self.bytes>self.limit and len(self.cache)>1:
                _,(_,n)=self.cache.popitem(last=False);self.bytes-=n
        self.cache.move_to_end(path)
        return self.cache[path][0]

    def sample(self,source,window):
        from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
        targets=set(window['target_blocks']);blocks=source['fixed5_blocks'];values=[]
        origin_prompt=self.tensor(source['clips'][0]['student_prompt_path'])
        clock_offset=origin_prompt['prompt_embeds'].shape[0]
        audio=self.tensor(source['audio_latents']);mask=self.tensor(source['audio_valid_mask'])
        for i in range(window['first_block'],window['last_block']+1):
            b=blocks[i];clip=source['clips'][b['clip']];lo,hi=b['local_start'],b['local_end']
            video=self.tensor(clip['video_latents'])[:,lo:hi]
            if video.shape!=(24,5,48,84) or not bool(torch.isfinite(video).all()):
                raise ValueError('Native fixed5 video differs from admitted geometry')
            valid=self.tensor(clip['video_valid_mask'])[lo:hi]
            if not bool(valid.all()):raise ValueError('Padding entered a supervised/context video block')
            emb=self.tensor(clip['student_prompt_path'])
            if emb['prompt']!=clip['text'] or not bool((emb['text_token_tags']==1).all()):
                raise ValueError('Prompt differs from the admitted current-shot condition')
            v=patchify_video_latents(video[None].float(),(1,2,2))
            # Media RoPE keeps the source's FIRST text span as a stable origin;
            # later prompt lengths cannot move already committed video keys.
            pts=clip['video_pts'];pos=torch.empty((5,24,42,3),dtype=torch.float32)
            for j,latent in enumerate(range(lo,hi)):
                nominal=17*(latent//5)+(0,1,5,9,13)[latent%5]
                pos[j,:,:,0]=clock_offset+40*(pts[nominal]-source['time_zero_sec'])
            pos[:,:,:,1]=torch.arange(24)[None,:,None]
            pos[:,:,:,2]=torch.arange(42)[None,None,:]
            a0,a1=b['audio_start'],b['audio_end']
            a=audio[:,:,a0:a1].permute(0,2,1).reshape(-1,32).float()
            apos=torch.zeros((2,a1-a0,3),dtype=torch.float32)
            apos[:,:,0]=clock_offset+40*(source['audio_origin_sec']-source['time_zero_sec'])+torch.arange(a0,a1)[None]
            apos[:,:,1]=torch.arange(2)[:,None]
            past=tuple(x['block'] for x in blocks[window['first_block']:i]
                       if x['start_sec']>=b['start_sec']-8.-1e-7 and x['end_sec']<=b['start_sec']+1e-7)
            values.append(dict(block=i,video=v,video_pos=pos.reshape(-1,3),
                prompt=emb['prompt_embeds'],audio=a,audio_pos=apos.reshape(-1,3),
                audio_mask=mask[a0:a1].repeat(2),past=past,target=i in targets,
                source_start=b['start_sec'],source_end=b['end_sec'],kind=b['kind']))
        return values


def pack(blocks,device,generator,*,strategy='A',clean_override=None,resample_u=None,video_noises=None,
         include_clean=True,clean_only=False):
    """Noisy targets never read their own clean copy or any future clean/text.

    For parallel TF, gradients may flow through clean-context computations
    WITHIN this optimizer update, as in ordinary packed teacher forcing. A
    sequential gradient oracle must retain that same context graph. Inference
    commits detached KV, and no training cache survives an optimizer update.
    """
    overrides=clean_override or {};splits=[];positions=[];tags=[];times=[]
    videos=[];audios=[];prompts=[];vi=[];ai=[];ti=[];outputs=[];offset=0;voffset=0;aoffset=0
    last=max(b['block'] for b in blocks);vt=[];at=[];vm=[];am=[]
    for b in blocks:
        block=b['block'];prompt=b['prompt'].to(device);n=len(prompt)
        splits.append(Split('text',block,offset,offset+n,b['past']))
        p=torch.zeros((n,3),device=device);p[:,0]=torch.arange(n,device=device)
        positions.append(p);tags.append(torch.ones(n,device=device,dtype=torch.long));times.append(torch.ones(n,device=device))
        ti.extend(range(offset,offset+n));prompts.append(prompt);offset+=n
        clean=b['video'].to(device,torch.float32)
        # Clean rows do not contain audio. Joint noisy AV targets are separate.
        if include_clean and (block<last or clean_only):
            value=overrides.get(block,clean).to(device,torch.float32);n=len(value)
            splits.append(Split('clean',block,offset,offset+n,b['past']))
            positions.append(b['video_pos'].to(device));tags.append(torch.zeros(n,device=device,dtype=torch.long));times.append(torch.ones(n,device=device))
            vi.extend(range(offset,offset+n));videos.append(value);vm.extend([False]*n);vt.append(torch.zeros_like(value));offset+=n;voffset+=n
        if clean_only or (not b['target'] and resample_u is None):
            continue
        epsilon=torch.randn(clean.shape,device=device,generator=generator) if video_noises is None else video_noises[block]
        if resample_u is None:
            u=torch.rand((),device=device,generator=generator).clamp(1e-4,1-1e-4)
        else:u=torch.as_tensor(resample_u,device=device)
        sv=12*u/(1+11*u);sa=3*u/(1+2*u);tv=1-sv;ta=1-sa
        value=(1-sv)*clean+sv*epsilon;n=len(value)
        # Resampling is video-only: no GT audio/future-codec audio may enter
        # the reconstructed history. Native H3 accepts an empty audio stream.
        audio=b['audio'].to(device) if resample_u is None else clean.new_empty((0,blocks[0]['audio'].shape[-1]))
        ea=torch.randn(audio.shape,device=device,generator=generator)
        avalue=(1-sa)*audio+sa*ea;na=len(audio)
        splits.append(Split('noisy',block,offset,offset+n+na,b['past']))
        positions.append(torch.cat((b['video_pos'].to(device),b['audio_pos'].to(device) if na else clean.new_empty((0,3)))))
        tags.append(torch.cat((torch.zeros(n,device=device,dtype=torch.long),torch.full((na,),2,device=device,dtype=torch.long))))
        times.append(torch.cat((torch.full((n,),float(tv),device=device),torch.full((na,),float(ta),device=device))))
        vi.extend(range(offset,offset+n));ai.extend(range(offset+n,offset+n+na))
        videos.append(value);audios.append(avalue);vt.append(clean-epsilon);at.append(audio-ea)
        vm.extend([True]*n);am.append(b['audio_mask'].to(device) if na else torch.empty(0,device=device,dtype=torch.bool))
        outputs.append(dict(block=block,video_slice=(voffset,voffset+n),audio_slice=(aoffset,aoffset+na),sigma_video=float(sv)))
        offset+=n+na;voffset+=n;aoffset+=na
    all_times=torch.cat(times);unique,tidx=torch.unique(all_times,sorted=True,return_inverse=True)
    d=blocks[0]['video'].shape[-1];ad=blocks[0]['audio'].shape[-1]
    inputs=dict(hidden_states=torch.cat(videos)[None] if videos else torch.empty((1,0,d),device=device),
        audio_hidden_states=torch.cat(audios)[None] if audios else torch.empty((1,0,ad),device=device),
        encoder_hidden_states=torch.cat(prompts)[None],position_ids=torch.cat(positions),
        token_tags=torch.cat(tags),timestep=unique,timestep_indices=tidx,
        text_indices=torch.tensor(ti,device=device,dtype=torch.long),video_indices=torch.tensor(vi,device=device,dtype=torch.long),
        audio_indices=torch.tensor(ai,device=device,dtype=torch.long),return_dict=False)
    return dict(inputs=inputs,context=Context(tuple(splits),strategy=='B'),outputs=outputs,
        video_target=torch.cat(vt) if vt else torch.empty((0,d),device=device),audio_target=torch.cat(at) if at else torch.empty((0,ad),device=device),
        video_mask=torch.tensor(vm,device=device,dtype=torch.bool),
        audio_mask=torch.cat(am) if am else torch.empty((0,),device=device,dtype=torch.bool))


def loss(output,batch):
    v,a=output;vm,am=batch['video_mask'],batch['audio_mask']
    if not bool(vm.any()) or not bool(am.any()):raise ValueError('Joint TF supervision is empty')
    lv=(v[0,vm].float()-batch['video_target'][vm]).square().mean()
    la=(a[0,am].float()-batch['audio_target'][am]).square().mean()
    if torch.distributed.is_initialized():
        counts=torch.stack((vm.sum(),am.sum())).to(device=v.device,dtype=torch.float64)
        global_counts=counts.clone();torch.distributed.all_reduce(global_counts)
        # FSDP averages gradients across ranks. Weight each local mean so the
        # result equals the global valid-row mean, including singleton tails.
        factors=counts*torch.distributed.get_world_size()/global_counts
        lv=lv*factors[0].float();la=la*factors[1].float()
    return lv+la,lv,la
