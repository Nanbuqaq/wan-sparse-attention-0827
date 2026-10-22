"""Short native-rule/shared-A and native/legacy-spatial diagnostic.

This is not a streaming benchmark. It uses a native legal 5.167s current
document, fixed noise across conditions, and Agent 5's actual A processor.
It never changes the production mask, training code or cached inference job.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

import torch
import torch.distributed as dist
from train_h3_v2_local_history_ddp import bootstrap
bootstrap()
from h3_shared_loader import load_session
from h3_shared_stage1 import shared_modules
from src.inference.render import generate_latents,decode_and_save


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class Proxy:
    def __init__(self,model,core,rule,grid):
        self.model,self.core,self.rule,self.grid=model,core,rule,grid
        self.calls=0;self.effect=None;self.received=None;self.first_text_time=None
    def __getattr__(self,name):
        try:return getattr(self.model,name)
        except AttributeError:return getattr(self.model.transformer,name)
    def __call__(self,**inputs):
        inputs=dict(inputs)
        if self.grid=='legacy':
            pos=inputs['position_ids'].clone();vi,ai=inputs['video_indices'],inputs['audio_indices']
            count=vi.numel()//1008
            frame=torch.stack(torch.meshgrid(torch.arange(24,device=pos.device),torch.arange(42,device=pos.device),indexing='ij'),dim=-1).reshape(1008,2)
            pos[vi,1:]=frame.repeat(count,1).to(pos.dtype)
            pos[ai,1]=torch.cat((torch.zeros(ai.numel()//2,device=pos.device),torch.ones(ai.numel()//2,device=pos.device)))
            pos[ai,2]=0
            inputs['position_ids']=pos
        if self.rule=='shared_A':
            rows=inputs['timestep'][inputs['timestep_indices']].clone()
            rows[inputs['text_indices']]=1.
            inputs['timestep'],inputs['timestep_indices']=torch.unique(rows,sorted=True,return_inverse=True)
            nt=inputs['text_indices'].numel();total=inputs['position_ids'].shape[0]
            ctx=self.core.Context((self.core.Split('text',0,0,nt,()),self.core.Split('noisy',0,nt,total,())))
            inputs['attention_mask']=self.core.bind(self.model.transformer,inputs,ctx)
        if self.calls==0:
            self.received=digest(inputs['encoder_hidden_states'])
            self.first_text_time=float(inputs['timestep'][inputs['timestep_indices'][inputs['text_indices'][0]]])
        output=self.model(**inputs)
        if self.calls==0:
            changed={**inputs,'encoder_hidden_states':torch.zeros_like(inputs['encoder_hidden_states'])}
            negative=self.model(**changed)
            self.effect=dict(video_relative_l2=float(torch.linalg.vector_norm(output[0].float()-negative[0].float())/
                             torch.linalg.vector_norm(output[0].float()).clamp_min(1e-12)),
                             audio_relative_l2=float(torch.linalg.vector_norm(output[1].float()-negative[1].float())/
                             torch.linalg.vector_norm(output[1].float()).clamp_min(1e-12)))
        self.calls+=1
        return output


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--request',required=True);args=ap.parse_args()
    request=json.loads(Path(args.request).read_text());rank,local=int(os.environ['RANK']),int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);torch.set_num_threads(2);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=20),device_id=device)
    out=Path(os.environ['INFER_OUTPUT_DIR'])
    try:
        if dist.get_world_size()!=8:raise ValueError('Eight independent conditions/seeds in one official job')
        rule,grid=[('native','native'),('shared_A','native'),('native','legacy'),('shared_A','legacy')][rank%4]
        seed=request['seed']+rank//4
        code,source=shared_modules();core=code['h3_fixed_kv_stage1']
        session=load_session(request['weights_spec'],device,dense=True)
        model=session['model']
        if rule=='shared_A':
            blocks=[]
            for wrapper in model.transformer.transformer_blocks:
                block=wrapper
                while hasattr(block,'_checkpoint_wrapped_module'):block=block._checkpoint_wrapped_module
                blocks.append(block)
            core.install(SimpleNamespace(transformer_blocks=blocks,token_refiner=model.transformer.token_refiner))
        proxy=Proxy(model,core,rule,grid)
        prompt=torch.load(request['prompt_path'],map_location='cpu',weights_only=True)
        expected=digest(prompt['prompt_embeds'])
        noise={}
        def capture(video,audio):noise.update(video=digest(video),audio=digest(audio))
        times=[];started=time.monotonic()
        video,audio=generate_latents(proxy,prompt['prompt_embeds'].to(device),prompt['text_token_tags'],124,50,seed,device,
            step_seconds=times,initial_noise_callback=capture)
        assert len(times)==50 and proxy.calls==50 and proxy.received==expected
        name=f'rank{rank}_{rule}_{grid}_seed{seed}'
        torch.save(dict(video=video.cpu(),audio=audio.cpu(),initial_noise=noise),out/(name+'_LATENTS.pt'))
        decode_and_save(video,audio,session['vae'],session['audio_vae'],str(out/(name+'.mp4')),str(device))
        report=dict(status='generated',rank=rank,rule=rule,spatial_grid=grid,seed=seed,NFE=50,
                    current_document_latent_frames=37,frames=124,duration_sec=124/24,
                    prompt=request['prompt'],prompt_embedding_sha256=expected,actual_model_text_sha256=proxy.received,
                    text_intervention_first_NFE=proxy.effect,first_text_timestep=proxy.first_text_time,
                    noise=noise,video=str(out/(name+'.mp4')),video_sha256=hashlib.sha256((out/(name+'.mp4')).read_bytes()).hexdigest(),
                    native_time_grid_and_sampler_used=True,shared_mask_source=source['bundle_sha256'],
                    native_vs_A_is_interface_recipe_not_single_mask_factor=True,
                    diagnostic_only=True,formal_benchmark_result=False,training_code_untouched=True,
                    total_seconds=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        (out/(name+'_RESULT.json')).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
        dist.barrier()
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
