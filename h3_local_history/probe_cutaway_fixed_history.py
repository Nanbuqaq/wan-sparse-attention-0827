#!/usr/bin/env python3
"""Replay one cutaway span with fixed old/new past and text-only mask changes."""
import argparse,datetime,hashlib,json,os,time,wave,subprocess
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor,distribute_tensor
from train_h3_v2_local_history_ddp import bootstrap,install_lora
bootstrap()
from h3_spatial_history_flow import HistoryReadAttention,_apply_rotary_emb
from h3_fsdp2_speed import shard_speed_model,apply_variant
from h3_fixed_chunk_contract import FixedTrainingModel,fixed_matched_batch
from h3_fixed_stream_decode import FixedChunkVideoDecoder
from h3_variable_sampling_case import decoder_chunks
from sample_demo import unpack_video
from probe_h3_fixed5_codec import video_file
import imageio_ffmpeg

class TextReadDiagnostic(HistoryReadAttention):
    def __call__(self,attn,x,rotary_emb=None,attention_mask=None,is_causal=False):
        if self.text_policy=='prefix_only':return super().__call__(attn,x,rotary_emb,attention_mask,is_causal)
        assert attention_mask is None and not is_causal
        from torch.nn.attention import SDPBackend,sdpa_kernel
        p=self.prefix_rows;t=self.text_rows;n=x.shape[1];assert 0<t<p<n
        k=attn.norm_k(attn.to_k(x).unflatten(-1,(attn.heads,-1)))
        if rotary_emb is not None:k=_apply_rotary_emb(k,*rotary_emb)
        k=k.transpose(1,2);v=attn.to_v(x).unflatten(-1,(attn.heads,-1)).transpose(1,2)
        kt,vt=k,v
        if self.text_policy=='text_current':
            kt=torch.cat((k[:,:,:t],k[:,:,p:]),dim=2);vt=torch.cat((v[:,:,:t],v[:,:,p:]),dim=2)
        result=[]
        for lo,hi,keys,values in [(0,t,kt,vt),(t,p,k[:,:,:p],v[:,:,:p]),(p,n,k,v)]:
            for start in range(lo,hi,self.query_chunk_size):
                end=min(hi,start+self.query_chunk_size)
                q=attn.norm_q(attn.to_q(x[:,start:end]).unflatten(-1,(attn.heads,-1)))
                if rotary_emb is not None:q=_apply_rotary_emb(q,rotary_emb[0][start:end],rotary_emb[1][start:end])
                q=q.transpose(1,2)
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):out=F.scaled_dot_product_attention(q,keys,values,dropout_p=0.)
                out=out.transpose(1,2).flatten(2,3);result.append(attn.to_out[1](attn.to_out[0](out)))
        return torch.cat(result,dim=1)


def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def main():
    from src.config.inference import InferenceConfig
    from src.inference.utils.assemble import build_inference_model
    from diffusers import AutoencoderKLMiniMaxH3,AutoencoderKLMiniMaxH3Audio
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--base-source',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK']);assert world==4
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(2)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=60))
    try:
        root=Path(a.input);meta=json.loads((root/'INPUT.json').read_text());data=torch.load(root/'FIXED_INPUTS.pt',map_location='cpu',weights_only=True)
        prompt=torch.load(root/'prompt.pt',map_location='cpu',weights_only=True);assert prompt['prompt']==meta['prompt']
        variants=[('old_past_prefix','old_past','prefix_only'),('new_past_prefix','new_past','prefix_only'),('new_past_text_full','new_past','text_full'),('new_past_text_current','new_past','text_current')]
        name,past_key,policy=variants[rank];out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
        torch.manual_seed(20261013);cfg=InferenceConfig();cfg.base_source=cfg.vae_source=a.base_source;cfg.kernels.inference_kernels=False
        load=time.monotonic();tr=build_inference_model(cfg,str(device),load_decoders=False,log=rank==0).transformer;tr.requires_grad_(False)
        processors=[]
        for block in tr.transformer_blocks:
            p=TextReadDiagnostic(8192,False);p.text_policy=policy;block.attn.set_processor(p);processors.append(p)
        install_lora(tr,8,8);model=FixedTrainingModel(tr).to(device);mesh=init_device_mesh('cuda',(world,))
        reshard=min(torch.cuda.get_device_properties(i).total_memory for i in range(world))<100*2**30
        model,blocks,wrappers=shard_speed_model(model,mesh,reshard_after_forward=reshard);apply_variant('no_inner_8k',blocks,wrappers,processors)
        saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);named=dict(model.named_parameters())
        with torch.no_grad():
            for key,value in saved['adapters'].items():
                p=named[key];p.copy_(distribute_tensor(value.to(device),mesh,p.placements) if isinstance(p,DTensor) else value.to(device))
        model.eval();torch.cuda.empty_cache();load_seconds=time.monotonic()-load
        source=meta['source'];row=meta['row'];start,end=row['video_latent_start'],row['video_latent_end'];aa,ae=row['audio_latent_start'],row['audio_latent_end']
        assert start==12 and end==22
        video=torch.zeros(24,source['video_latent_count'],48,84,dtype=torch.bfloat16);audio=torch.zeros(2,32,source['audio_latent_count'],dtype=torch.bfloat16)
        mask=torch.ones(source['audio_latent_count'],dtype=torch.bool);gen=torch.Generator(device=device).manual_seed(20261016)
        past=data[past_key];assert past.shape[1]==12
        batch=fixed_matched_batch(row,source,video,audio,prompt['prompt_embeds'],prompt['text_token_tags'],mask,gen,device,retained_ids=list(range(12)),retained_video=past)
        batch.update(sampling=True,zero_history_state=False);inputs=batch['inputs']
        for p in processors:p.prefix_rows=batch['prefix_rows'];p.text_rows=inputs['text_indices'].numel()
        xv=data['initial_video_noise'].to(device);xa=data['initial_audio_noise'].to(device)
        assert xv.shape==inputs['hidden_states'].shape and xa.shape==inputs['audio_hidden_states'].shape
        record=dict(variant=name,text_query_policy=policy,past_key=past_key,past_sha256=hashlib.sha256(past.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
            prompt_sha256=meta['prompt_sha256'],video_noise_sha256=digest(xv),audio_noise_sha256=digest(xa),positions_sha256=digest(inputs['position_ids']),
            checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),history_ids=list(range(12)),source_time_window=[row['source_target_start_sec'],row['source_target_end_sec']],
            current_text_rows=int(inputs['text_indices'].numel()),history_rows=int(batch['history_rows'].shape[1]),
            history_queries='text+history prefix only',current_queries='whole document',source_video_or_audio_read=False,GPU=torch.cuda.get_device_name(device))
        (out/f'{name}_RESOLVED.json').write_text(json.dumps(record,indent=2)+'\n')
        torch.cuda.reset_peak_memory_stats();dist.barrier();tick=time.monotonic()
        with torch.inference_mode():
            for i in range(50):
                u=1-i/50;unext=1-(i+1)/50;tv=1-12*u/(1+11*u);ta=1-3*u/(1+2*u);vn=1-12*unext/(1+11*unext);an=1-3*unext/(1+2*unext)
                rt=torch.full((len(inputs['position_ids']),),tv,device=device);rt[inputs['state_indices']]=1.;rt[inputs['audio_indices']]=ta
                inputs['timestep'],inputs['timestep_indices']=torch.unique(rt,sorted=True,return_inverse=True);inputs['hidden_states']=xv;inputs['audio_hidden_states']=xa
                pv,pa=model(batch);xv=xv+(vn-tv)*pv.float();xa=xa+(an-ta)*pa.float()
                if i in [0,24,49]:torch.cuda.synchronize();print(json.dumps(dict(variant=name,NFE_complete=i+1,seconds=time.monotonic()-tick)),flush=True)
            torch.cuda.synchronize();dit=time.monotonic()-tick;assert torch.isfinite(xv).all() and torch.isfinite(xa).all()
            z=unpack_video(xv,end-start,48,84).to(torch.bfloat16)
            ref=data['old_reference' if past_key=='old_past' else 'new_reference'];err=z[0].cpu().float()-ref.float()
            torch.save(dict(video=z.cpu(),audio=xa.cpu(),contract=record),out/f'{name}_GENERATED.pt')
            vae=AutoencoderKLMiniMaxH3.from_pretrained(a.base_source,subfolder='vae').to(device).eval();decoder=FixedChunkVideoDecoder(vae,device)
            pieces=[decoder.push(x).cpu() for x in decoder_chunks(z,start)];pieces.append(decoder.flush().cpu());pixels=torch.cat(pieces,dim=2);assert pixels.shape[2]==34
            path=out/f'{name}.mp4';artifact=video_file(path,pixels,24.);vae.to('cpu');torch.cuda.empty_cache()
            avae=AutoencoderKLMiniMaxH3Audio.from_pretrained(a.base_source,subfolder='audio_vae').to(device).eval()
            az=xa[0].reshape(2,ae-aa,32).permute(0,2,1).to(torch.bfloat16);mean=torch.tensor(avae.config.latents_mean,device=device).view(1,-1,1);std=torch.tensor(avae.config.latents_std,device=device).view(1,-1,1)
            aw=avae.decode(az*std+mean,return_dict=False)[0].float().permute(1,0,2)[0].clamp(-1,1).cpu();wav=path.with_suffix('.wav')
            with wave.open(str(wav),'wb') as f:f.setnchannels(2);f.setsampwidth(2);f.setframerate(32000);f.writeframes((aw.T*32767).round().short().contiguous().numpy().tobytes())
            mux=out/f'{name}.mux.mp4';subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-nostdin','-y','-i',str(path),'-i',str(wav),'-c:v','copy','-c:a','aac','-af','apad','-t',str(34/24),str(mux)],check=True);os.replace(mux,path);artifact['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        record.update(artifact=artifact,reference_max_abs=float(err.abs().max()),reference_mse=float(err.square().mean()),DiT_seconds=dit,load_seconds=load_seconds,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,quality_pass=None,training_changed=False,default_policy_changed=False)
        (out/f'{name}_RESULT.json').write_text(json.dumps(record,indent=2)+'\n');dist.barrier()
        if rank==0:
            all_results=[json.loads((out/f'{x[0]}_RESULT.json').read_text()) for x in variants]
            for key in ['prompt_sha256','video_noise_sha256','audio_noise_sha256','positions_sha256','checkpoint_sha256']:
                assert len({r[key] for r in all_results})==1,key
            assert len({r['past_sha256'] for r in all_results[1:]})==1
            (out/'CONTROL_PASS.json').write_text(json.dumps(dict(videos=4,same_weights_noise_prompt_positions=True,same_new_history_for_mask_variants=True,semantic_review_pending=True),indent=2)+'\n')
    finally:dist.destroy_process_group()

if __name__=='__main__':main()
