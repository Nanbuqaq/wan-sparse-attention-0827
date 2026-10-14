#!/usr/bin/env python3
"""FSDP2 H3 sampling: generated spatial history, same-shot decode, hard cuts."""
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
from torch.distributed.tensor import DTensor,distribute_tensor
from train_h3_v2_local_history_ddp import bootstrap,install_lora
from h3_fsdp2_speed import shard_speed_model,apply_variant
from h3_fixed_chunk_contract import FixedTrainingModel,fixed_matched_batch,history_ids
from h3_fixed_stream_decode import FixedChunkVideoDecoder
from h3_generated_video_history import GeneratedVideoHistory,wrong_past_video
from h3_variable_sampling_case import annotation_variable_cases,decoder_chunks
from probe_h3_fixed5_codec import video_file


def unpack_video(rows,frames,h,w):
    return rows.reshape(1,frames,h//2,w//2,24,1,2,2).permute(0,4,1,5,2,6,3,7).reshape(1,24,frames,h,w)


def main():
    bootstrap()
    from diffusers import AutoencoderKLMiniMaxH3,AutoencoderKLMiniMaxH3Audio
    from h3_spatial_history_flow import install_history_attention
    from src.config.inference import InferenceConfig
    from src.inference.utils.assemble import build_inference_model
    ap=argparse.ArgumentParser();ap.add_argument('--base-source',required=True)
    ap.add_argument('--checkpoint',required=True);ap.add_argument('--cases',required=True)
    ap.add_argument('--source-contracts',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--nfe',type=int,default=50);ap.add_argument('--label',required=True)
    ap.add_argument('--comparison-mode',choices=('zero','reset','wrong'),default='zero')
    ap.add_argument('--wrong-history-root')
    ap.add_argument('--sampling-protocol',choices=('fixed','annotation_variable'),default='fixed')
    ap.add_argument('--adapter-mode',choices=('trained','native_init'),default='trained')
    args=ap.parse_args();rank,world,local=(int(os.environ[k]) for k in ('RANK','WORLD_SIZE','LOCAL_RANK'))
    assert world==8;torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(2)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=30))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    try:
        sources=json.loads(Path(args.source_contracts).read_text());cases=json.loads(Path(args.cases).read_text())['cases']
        if args.sampling_protocol=='annotation_variable':
            if args.comparison_mode=='wrong':raise ValueError('Wrong peer control currently uses the fixed sampling protocol.')
            cases=annotation_variable_cases(cases)
        case=cases[rank//2];mode='carry' if rank%2==0 else args.comparison_mode;source=sources[case['source_id']]
        peer_case=cases[(rank//2+1)%4];peer=None
        if mode=='wrong':
            if not args.wrong_history_root:raise ValueError('Wrong history requires a previously generated peer archive.')
            peer_rank=2*((rank//2+1)%4)
            peer=torch.load(Path(args.wrong_history_root)/f'rank{peer_rank}_generated_latents.pt',map_location='cpu',weights_only=False)
            peer_result=json.loads((Path(args.wrong_history_root)/f'rank{peer_rank}_RESULT.json').read_text())
            assert peer['case_id']==peer_case['case_id'] and not peer_result['clean_gt_latents_read']
            assert peer_result['checkpoint_sha256']==hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
        torch.manual_seed(20261013);cfg=InferenceConfig();cfg.base_source=cfg.vae_source=args.base_source;cfg.kernels.inference_kernels=False
        start_load=time.monotonic();transformer=build_inference_model(cfg,str(device),load_decoders=False,log=rank==0).transformer
        transformer.requires_grad_(False);processors=install_history_attention(transformer,8192,False);install_lora(transformer,8,8)
        model=FixedTrainingModel(transformer).to(device);mesh=init_device_mesh('cuda',(world,))
        inference_reshard=min(torch.cuda.get_device_properties(i).total_memory for i in range(8))<100*2**30
        model,blocks,wrappers=shard_speed_model(model,mesh,reshard_after_forward=inference_reshard);apply_variant('no_inner_8k',blocks,wrappers,processors)
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False);named=dict(model.named_parameters())
        if args.adapter_mode=='native_init':
            saved=dict(saved,step=0,adapters={})
            assert all(torch.count_nonzero(p).item()==0 for name,p in named.items() if name.endswith('lora_b'))
        with torch.no_grad():
            for name,value in saved['adapters'].items():
                p=named[name];p.copy_(distribute_tensor(value.to(device),mesh,p.placements) if isinstance(p,DTensor) else value.to(device))
        model.eval();torch.cuda.empty_cache()
        vae=AutoencoderKLMiniMaxH3.from_pretrained(args.base_source,subfolder='vae').to(device).eval()
        audio_vae=AutoencoderKLMiniMaxH3Audio.from_pretrained(args.base_source,subfolder='audio_vae').to(device).eval()
        decoder=FixedChunkVideoDecoder(vae,device);load_seconds=time.monotonic()-start_load
        h,w=48,84;video=torch.zeros(24,source['video_latent_count'],h,w,dtype=torch.bfloat16)
        audio=torch.zeros(2,32,source['audio_latent_count'],dtype=torch.bfloat16)
        # Shape/validity only; no video.pt or audio.pt is read by this sampler.
        mask=torch.ones(source['audio_latent_count'],dtype=torch.bool)
        generator=torch.Generator(device=device).manual_seed(20261014+rank//2)
        generated_ids=set();active_history=GeneratedVideoHistory(source)
        pixels=[];audio_pieces=[];records=[];parent=None
        torch.cuda.reset_peak_memory_stats();dist.barrier()
        with torch.inference_mode():
          for number,row in enumerate(case['chunks']):
            chunk_start=time.monotonic();prompt=torch.load(row['prompt_path'],map_location='cpu',weights_only=True)
            if mode=='reset':active_history.reset()
            chosen,history_video=active_history.select(row['video_latent_start'])
            peer_ids=[]
            if mode=='wrong' and chosen:
                history_video,peer_ids=wrong_past_video(peer['video'],peer_case['chunks'],number,len(chosen))
            batch=fixed_matched_batch(row,source,video,audio,prompt['prompt_embeds'],prompt['text_token_tags'],mask,
                generator,device,retained_ids=chosen,retained_video=history_video)
            batch.update(sampling=True,zero_history_state=mode=='zero')
            for processor in processors:processor.prefix_rows=batch['prefix_rows']
            inputs=batch['inputs'];noise_hash=hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest()
            xvideo=torch.randn(inputs['hidden_states'].shape,generator=generator,device=device)
            xaudio=torch.randn(inputs['audio_hidden_states'].shape,generator=generator,device=device)
            video_noise_sha256=hashlib.sha256(xvideo.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            audio_noise_sha256=hashlib.sha256(xaudio.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            denoise_start=time.monotonic()
            for i in range(args.nfe):
                u=1-i/args.nfe;unext=1-(i+1)/args.nfe
                tv=1-12*u/(1+11*u);ta=1-3*u/(1+2*u)
                tvnext=1-12*unext/(1+11*unext);tanext=1-3*unext/(1+2*unext)
                row_t=torch.full((len(inputs['position_ids']),),tv,device=device)
                row_t[inputs['state_indices']]=1.;row_t[inputs['audio_indices']]=ta
                inputs['timestep'],inputs['timestep_indices']=torch.unique(row_t,sorted=True,return_inverse=True)
                inputs['hidden_states']=xvideo;inputs['audio_hidden_states']=xaudio
                pv,pa=model(batch)
                xvideo=xvideo+(tvnext-tv)*pv.float();xaudio=xaudio+(tanext-ta)*pa.float()
            torch.cuda.synchronize();denoise_seconds=time.monotonic()-denoise_start
            if not torch.isfinite(xvideo).all() or not torch.isfinite(xaudio).all():raise RuntimeError('Nonfinite generated chunk')
            a,b=row['video_latent_start'],row['video_latent_end'];aa,ae=row['audio_latent_start'],row['audio_latent_end']
            chunk=unpack_video(xvideo,b-a,h,w).to(torch.bfloat16)
            video[:,a:b]=chunk[0].cpu();generated_ids.update(range(a,b))
            active_history.put(a,chunk[0]);evicted=active_history.evict(b)
            gen_audio=xaudio[0].reshape(2,ae-aa,32).permute(0,2,1).to(torch.bfloat16)
            audio_pieces.append(gen_audio);audio[:,:,aa:ae]=gen_audio.cpu()
            decode_start=time.monotonic()
            cut=parent is not None and (parent!=row['parent_window_id'] or previous_end!=a)
            if cut:
                tail=decoder.flush()
                if tail is not None:pixels.append(tail.cpu())
            for part in decoder_chunks(chunk,a):pixels.append(decoder.push(part).cpu())
            torch.cuda.synchronize();decode_seconds=time.monotonic()-decode_start
            parent=row['parent_window_id'];previous_end=b
            record=dict(rank=rank,chunk=number,window_id=row['window_id'],case_id=case['case_id'],state_mode=mode,
                history_frame_ids=chosen,history_source='independent_generated_peer_past' if mode=='wrong' and chosen else 'own_generated_video_only',clean_gt_history_used=False,
                wrong_history_source=peer_case['source_id'] if peer_ids else None,wrong_history_frame_ids=peer_ids,
                active_history_bytes=active_history.bytes,evicted_frame_ids=evicted,
                sampling_protocol=args.sampling_protocol,generated_target_latents=b-a,
                current_prompt_sha256=hashlib.sha256(Path(row['prompt_path']).read_bytes()).hexdigest(),
                noise_state_sha256=noise_hash,video_noise_sha256=video_noise_sha256,audio_noise_sha256=audio_noise_sha256,source_target_start_sec=row['source_target_start_sec'],source_target_end_sec=row['source_target_end_sec'],
                hard_cut_decoder_reset=cut,model_history_reset=mode=='reset',nfe=args.nfe,cfg_scale=1.,
                denoise_seconds=denoise_seconds,video_decode_seconds=decode_seconds,chunk_wall_seconds=time.monotonic()-chunk_start)
            records.append(record);print(json.dumps(record),flush=True)
          pixels.append(decoder.flush().cpu())
          decoded=torch.cat(pixels,dim=2)
          path=out/f"{args.label}_{case['source_id']}_{mode}.mp4";encoding_start=time.monotonic()
          artifact=video_file(path,decoded,case['nominal_output_fps'])
          # Audio is an offline complete-sequence decode, not a claimed online codec.
          al=torch.cat(audio_pieces,dim=2);mean=torch.tensor(audio_vae.config.latents_mean,device=device).view(1,-1,1)
          std=torch.tensor(audio_vae.config.latents_std,device=device).view(1,-1,1)
          aw=audio_vae.decode(al*std+mean,return_dict=False)[0].float().permute(1,0,2)[0].clamp(-1,1).cpu()
          wav=path.with_suffix('.wav');pcm=(aw.T*32767).round().short().contiguous().numpy().tobytes()
          with wave.open(str(wav),'wb') as f:f.setnchannels(2);f.setsampwidth(2);f.setframerate(32000);f.writeframes(pcm)
          torch.save(dict(video=video,audio=audio,generated_ids=sorted(generated_ids),case_id=case['case_id']),out/f'rank{rank}_generated_latents.pt')
          mux=path.with_name(path.stem+'.mux.mp4')
          subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-nostdin','-y','-i',str(path),'-i',str(wav),
              '-c:v','copy','-c:a','aac','-af','apad','-t',str(decoded.shape[2]/case['nominal_output_fps']),str(mux)],check=True)
          os.replace(mux,path)
          artifact.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),audio_muxed=True)
        result=dict(adapter_mode=args.adapter_mode,checkpoint_adapters_applied=args.adapter_mode=='trained',adapter_initialization='base proj_in copy + LoRA B=0' if args.adapter_mode=='native_init' else None,
            hardware_host=__import__('socket').gethostname(),hardware_gpu=torch.cuda.get_device_name(device),torch_version=torch.__version__,
            comparison_track='same_backbone_h3_engineering_diagnostic',
            audio_future_context_allowed=False,audio_encoder_lookahead_sec=0.,source_audio_encoder_lookahead_sec=0.310125,
            audio_valid_mask_after_encoder='N/A: no source audio encode',formal_av_alignment_status='missing',
            result_kind='two short annotation shots, not long memory benchmark',status='generated_history_video_pass',rank=rank,case_id=case['case_id'],source_id=case['source_id'],
            checkpoint=args.checkpoint,checkpoint_step=saved['step'],checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            state_mode=mode,comparison_mode=args.comparison_mode,sampling_protocol=args.sampling_protocol,
            generation_chunk_count=len(records),records=records,artifact=artifact,audio_wav=str(wav),
            nfe=args.nfe,cfg_scale=1.,video_generation_and_native_stream_decode=True,
            clean_gt_latents_read=False,event_eval_ready=False,formal_memory_quality_claim=False,
            load_seconds=load_seconds,inference_parameter_resharding=inference_reshard,encode_and_audio_decode_seconds=time.monotonic()-encoding_start,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,offline_audio_decode_only=True)
        torch.save(dict(video=video,audio=audio,generated_ids=sorted(generated_ids),case_id=case['case_id']),out/f'rank{rank}_generated_latents.pt')
        (out/f'rank{rank}_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');dist.barrier()
        if rank==0:
            all_results=[json.loads((out/f'rank{r}_RESULT.json').read_text()) for r in range(8)]
            for c in range(4):
                pair=all_results[c*2:c*2+2]
                assert [r['noise_state_sha256'] for r in pair[0]['records']]==[r['noise_state_sha256'] for r in pair[1]['records']]
            (out/'GENERATION_PASS.json').write_text(json.dumps(dict(status='8_gpu_4_independent_sources_controlled_stream_generation_pass',
                videos=8,independent_sources=4,comparison_mode=args.comparison_mode,generated_history_only=True,
                sampling_protocol=args.sampling_protocol,generation_chunks_per_video=len(case['chunks']),
                active_history_physically_bounded=True,matched_current_noise_per_pair=True,event_eval_ready=False),indent=2)+'\n')
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
