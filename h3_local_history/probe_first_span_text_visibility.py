#!/usr/bin/env python3
"""One replayed empty-history span. Only prefix attention visibility differs."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import wave
import imageio_ffmpeg
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from train_h3_v2_local_history_ddp import bootstrap,install_lora
from h3_fsdp2_speed import shard_speed_model,apply_variant
from h3_fixed_chunk_contract import FixedTrainingModel,fixed_matched_batch
from h3_fixed_stream_decode import FixedChunkVideoDecoder
from h3_variable_sampling_case import decoder_chunks
from probe_h3_fixed5_codec import video_file
from sample_demo import unpack_video


def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    bootstrap()
    from h3_spatial_history_flow import install_history_attention
    from src.config.inference import InferenceConfig
    from src.inference.utils.assemble import build_inference_model
    from diffusers import AutoencoderKLMiniMaxH3,AutoencoderKLMiniMaxH3Audio
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--base-source',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK']);assert world==2
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(2)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=45))
    try:
        root=Path(a.input);manifest=json.loads((root/'INPUT.json').read_text());case=manifest['case'];source=manifest['source'];row=case['chunks'][0]
        assert row['video_latent_start']==0 and row['audio_latent_start']==0
        replay=torch.load(root/'replay.pt',map_location='cpu',weights_only=False)
        prompt=torch.load(root/'prompt.pt',map_location='cpu',weights_only=True)
        assert prompt['prompt']==manifest['prompt']
        out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
        policy='prefix_only' if rank==0 else 'full_document'
        torch.manual_seed(20261013);cfg=InferenceConfig();cfg.base_source=cfg.vae_source=a.base_source;cfg.kernels.inference_kernels=False
        load_start=time.monotonic();transformer=build_inference_model(cfg,str(device),load_decoders=False,log=rank==0).transformer
        transformer.requires_grad_(False);processors=install_history_attention(transformer,8192,False);install_lora(transformer,8,8)
        model=FixedTrainingModel(transformer).to(device);mesh=init_device_mesh('cuda',(world,))
        reshard=min(torch.cuda.get_device_properties(i).total_memory for i in range(world))<100*2**30
        model,blocks,wrappers=shard_speed_model(model,mesh,reshard_after_forward=reshard);apply_variant('no_inner_8k',blocks,wrappers,processors)
        model.eval();assert all(torch.count_nonzero(p).item()==0 for n,p in model.named_parameters() if n.endswith('lora_b'))
        torch.cuda.empty_cache();load_seconds=time.monotonic()-load_start
        v=torch.zeros(24,source['video_latent_count'],48,84,dtype=torch.bfloat16)
        au=torch.zeros(2,32,source['audio_latent_count'],dtype=torch.bfloat16)
        mask=torch.ones(source['audio_latent_count'],dtype=torch.bool);gen=torch.Generator(device=device).manual_seed(int(case['seed']))
        batch=fixed_matched_batch(row,source,v,au,prompt['prompt_embeds'],prompt['text_token_tags'],mask,gen,device,retained_ids=[])
        batch.update(sampling=True,zero_history_state=False)
        assert batch['history_rows'].shape[1]==0
        for p in processors:p.prefix_rows=batch['prefix_rows'];p.prefix_reads_current=rank==1
        inputs=batch['inputs'];xv=replay['initial_video_noise'].to(device);xa=replay['initial_audio_noise'].to(device)
        assert xv.shape==inputs['hidden_states'].shape and xa.shape==inputs['audio_hidden_states'].shape
        assert digest(xv)==manifest['video_noise_sha256'] and digest(xa)==manifest['audio_noise_sha256']
        contract=dict(policy=policy,source_snapshot_sha256=manifest['snapshot_sha256'],prompt_sha256=manifest['prompt_sha256'],
            video_noise_sha256=digest(xv),audio_noise_sha256=digest(xa),history_rows=0,nfe=50,cfg=1.,
            position_ids_sha256=digest(inputs['position_ids']),token_tags_sha256=digest(inputs['token_tags']),
            prompt_embedding_sha256=hashlib.sha256(prompt['prompt_embeds'].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
            model_weights='released Dense H3; base projection copy; zero LoRA B; no trained adapters',
            GPU=torch.cuda.get_device_name(device),host=__import__('socket').gethostname(),reshard_after_forward=reshard)
        (out/f'{policy}_RESOLVED.json').write_text(json.dumps(contract,indent=2)+'\n')
        torch.cuda.reset_peak_memory_stats();dist.barrier();start=time.monotonic();steps=[]
        with torch.inference_mode():
            for i in range(50):
                tick=time.monotonic();u=1-i/50;unext=1-(i+1)/50
                tv=1-12*u/(1+11*u);ta=1-3*u/(1+2*u);vn=1-12*unext/(1+11*unext);an=1-3*unext/(1+2*unext)
                rt=torch.full((len(inputs['position_ids']),),tv,device=device);rt[inputs['audio_indices']]=ta
                inputs['timestep'],inputs['timestep_indices']=torch.unique(rt,sorted=True,return_inverse=True)
                inputs['hidden_states']=xv;inputs['audio_hidden_states']=xa
                pv,pa=model(batch);xv=xv+(vn-tv)*pv.float();xa=xa+(an-ta)*pa.float()
                torch.cuda.synchronize();steps.append(time.monotonic()-tick)
                if i in [0,24,49]:print(json.dumps(dict(policy=policy,NFE_completed=i+1,seconds=time.monotonic()-start)),flush=True)
            assert torch.isfinite(xv).all() and torch.isfinite(xa).all()
            dit_seconds=time.monotonic()-start
            z=unpack_video(xv,row['video_latent_end'],48,84).to(torch.bfloat16)
            reference=replay['video'][:,:row['video_latent_end']].float();difference=z[0].cpu().float()-reference
            torch.save(dict(video=z.cpu(),audio=xa.cpu(),contract=contract),out/f'{policy}_GENERATED.pt')
            vae=AutoencoderKLMiniMaxH3.from_pretrained(a.base_source,subfolder='vae').to(device).eval()
            decoder=FixedChunkVideoDecoder(vae,device);pixels=[decoder.push(part).cpu() for part in decoder_chunks(z,0)];pixels.append(decoder.flush().cpu())
            decoded=torch.cat(pixels,dim=2);assert decoded.shape[2]==39
            path=out/f'{policy}.mp4';artifact=video_file(path,decoded,24.);vae.to('cpu');torch.cuda.empty_cache()
            avae=AutoencoderKLMiniMaxH3Audio.from_pretrained(a.base_source,subfolder='audio_vae').to(device).eval()
            za=xa[0].reshape(2,row['audio_latent_end'],32).permute(0,2,1).to(torch.bfloat16)
            mean=torch.tensor(avae.config.latents_mean,device=device).view(1,-1,1);std=torch.tensor(avae.config.latents_std,device=device).view(1,-1,1)
            wav=path.with_suffix('.wav');aw=avae.decode(za*std+mean,return_dict=False)[0].float().permute(1,0,2)[0].clamp(-1,1).cpu()
            with wave.open(str(wav),'wb') as f:f.setnchannels(2);f.setsampwidth(2);f.setframerate(32000);f.writeframes((aw.T*32767).round().short().contiguous().numpy().tobytes())
            mux=out/f'{policy}.mux.mp4';subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-nostdin','-y','-i',str(path),'-i',str(wav),'-c:v','copy','-c:a','aac','-af','apad','-t',str(39/24),str(mux)],check=True);os.replace(mux,path)
            artifact['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        result=dict(**contract,artifact=artifact,load_seconds=load_seconds,DiT_seconds=dit_seconds,per_NFE_seconds=steps,
            original_prefix_replay_max_abs=float(difference.abs().max()),original_prefix_replay_mse=float(difference.square().mean()),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,quality_pass=None,strict_AV_alignment_pass=None,
            audio_encoder_lookahead_sec=0.,audio_future_context_allowed=False,audio_valid_mask_after_encoder='N/A no source audio encoding',
            generated_history_training=False,formal_benchmark=False,scope='single empty-history text-visibility diagnostic; no default policy changed')
        (out/f'{policy}_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');dist.barrier()
        if rank==0:
            pair=[json.loads((out/f'{p}_RESULT.json').read_text()) for p in ['prefix_only','full_document']]
            for key in ['video_noise_sha256','audio_noise_sha256','position_ids_sha256','token_tags_sha256','prompt_embedding_sha256']:assert pair[0][key]==pair[1][key]
            (out/'PROBE_INTERFACE_PASS.json').write_text(json.dumps(dict(videos=2,same_actual_noise=True,same_positions=True,same_model_weights=True,
                only_changed_condition='prefix query key visibility; history is empty',semantic_result_pending=True),indent=2)+'\n')
    finally:dist.destroy_process_group()

if __name__=='__main__':main()
