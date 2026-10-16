#!/usr/bin/env python3
"""Prompt/count demo: trained history, untrained same interface, native independent shots."""
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

class NativeDemoModel(FixedTrainingModel):
    @property
    def config(self):return self.transformer.config
    @property
    def transformer_blocks(self):return self.transformer.transformer_blocks
    def forward(self,*args,**kwargs):return self.transformer(*args,**kwargs)



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
    ap.add_argument('--sampling-protocol',choices=('full_annotation',),default='full_annotation')
    ap.add_argument('--collective-timeout-min',type=int,default=240)
    ap.add_argument('--noise-guard')
    ap.add_argument('--adapter-mode',choices=('trained','native_init','native_stock'),default='trained')
    ap.add_argument('--empty-history-attention',choices=('prefix_only','full_document'),default='prefix_only')
    ap.add_argument('--text-read-policy',choices=('legacy_prefix','text_full'),default='legacy_prefix')
    args=ap.parse_args();rank,world,local=(int(os.environ[k]) for k in ('RANK','WORLD_SIZE','LOCAL_RANK'))
    assert world in (2,4,8);torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(2)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=args.collective_timeout_min))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    try:
        sources=json.loads(Path(args.source_contracts).read_text());cases=json.loads(Path(args.cases).read_text())['cases']
        if args.sampling_protocol=='annotation_variable':
            if args.comparison_mode=='wrong':raise ValueError('Wrong peer control currently uses the fixed sampling protocol.')
            cases=annotation_variable_cases(cases)
        if args.comparison_mode=='wrong':raise ValueError('Full-annotation wrong-history replay is not implemented;do not relabel zero/reset as wrong.')
        assert len(cases)==world//2
        case=cases[rank//2];mode=('native' if args.adapter_mode=='native_stock' else ('carry' if rank%2==0 else args.comparison_mode));source=sources[case['source_id']]
        if args.adapter_mode=='native_stock':
            case=dict(case,seed=int(case['seed'])+rank%2)
        if not case.get('generation_demo_only'):raise ValueError('Demo requires an explicit synthetic generation-clock case')
        peer_case=cases[(rank//2+1)%len(cases)];peer=None
        if mode=='wrong':
            if not args.wrong_history_root:raise ValueError('Wrong history requires a previously generated peer archive.')
            peer_rank=2*((rank//2+1)%len(cases))
            peer=torch.load(Path(args.wrong_history_root)/f'rank{peer_rank}_generated_latents.pt',map_location='cpu',weights_only=False)
            peer_result=json.loads((Path(args.wrong_history_root)/f'rank{peer_rank}_RESULT.json').read_text())
            assert peer['case_id']==peer_case['case_id'] and not peer_result['clean_gt_latents_read']
            assert peer_result['checkpoint_sha256']==hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
        torch.manual_seed(20261013);cfg=InferenceConfig();cfg.base_source=cfg.vae_source=args.base_source;cfg.kernels.inference_kernels=False
        guards=json.loads(Path(args.noise_guard).read_text()) if args.noise_guard else {}
        expected_noise=guards.get(case['case_id'],{})
        if expected_noise:
            probe=torch.Generator(device=device).manual_seed(int(case['seed']))
            checks=[]
            for number,row in enumerate(case['chunks']):
                frames=row['video_latent_end']-row['video_latent_start'];ac=row['audio_latent_end']-row['audio_latent_start']
                torch.randn((1,24,frames,48,84),generator=probe,device=device)
                torch.randn((2*ac,32),generator=probe,device=device)
                torch.rand((),generator=probe,device=device)
                state_sha=hashlib.sha256(probe.get_state().cpu().numpy().tobytes()).hexdigest()
                nv=torch.randn((1,frames*1008,96),generator=probe,device=device)
                na=torch.randn((1,2*ac,32),generator=probe,device=device)
                vsha=hashlib.sha256(nv.cpu().contiguous().numpy().tobytes()).hexdigest()
                asha=hashlib.sha256(na.cpu().contiguous().numpy().tobytes()).hexdigest()
                if str(number) in expected_noise:
                    e=expected_noise[str(number)]
                    assert (state_sha,vsha,asha)==(e['noise_state_sha256'],e['video_noise_sha256'],e['audio_noise_sha256']),'existing actual noise must match before NN;never rely on seed across hardware'
                checks.append(dict(chunk=number,noise_state_sha256=state_sha,video_noise_sha256=vsha,audio_noise_sha256=asha))
            (out/f'rank{rank}_EXISTING_NOISE_GUARD_PASS.json').write_text(json.dumps(checks,indent=2)+'\n')
            del nv,na,probe
        resolved_config_path=out/f'rank{rank}_RESOLVED_CONFIG.json'
        resolved=dict(model_assembly_config=__import__('dataclasses').asdict(cfg),sampler_args=vars(args),case=case,source_contract=source,
            actual_sampling_NFE=args.nfe,actual_CFG=1.,actual_valid_video_frames=source['valid_output_frames'],world_size=world,
            render_config_scope='assembly render defaults are not read by this sampler;case windows and NFE/seed arguments define actual generation',
            GPU=torch.cuda.get_device_name(device),torch_version=torch.__version__)
        resolved_config_path.write_text(json.dumps(resolved,default=str,indent=2)+'\n')
        start_load=time.monotonic();transformer=build_inference_model(cfg,str(device),load_decoders=False,log=rank==0).transformer
        transformer.requires_grad_(False)
        if args.adapter_mode=='native_stock':
            processors=[];model=NativeDemoModel(transformer).to(device)
            for block in transformer.transformer_blocks:block.ff_chunk_size=8192;block.ff_chunk_checkpoint=False
        else:
            if args.text_read_policy=='text_full':
                from h3_prompt_history_attention import install_prompt_history_attention
                processors=install_prompt_history_attention(transformer,8192)
            else:processors=install_history_attention(transformer,8192,False)
            install_lora(transformer,8,8)
            model=FixedTrainingModel(transformer).to(device)
        mesh=init_device_mesh('cuda',(world,))
        inference_reshard=min(torch.cuda.get_device_properties(i).total_memory for i in range(world))<100*2**30
        model,blocks,wrappers=shard_speed_model(model,mesh,reshard_after_forward=inference_reshard);apply_variant('no_inner_8k',blocks,wrappers,processors)
        saved=({'step':0,'adapters':{}} if args.adapter_mode=='native_stock' else torch.load(args.checkpoint,map_location='cpu',weights_only=False));named=dict(model.named_parameters())
        if args.adapter_mode in ('native_init','native_stock'):
            saved=dict(saved,step=0,adapters={})
            assert all(torch.count_nonzero(p).item()==0 for name,p in named.items() if name.endswith('lora_b'))
        with torch.no_grad():
            for name,value in saved['adapters'].items():
                p=named[name];p.copy_(distribute_tensor(value.to(device),mesh,p.placements) if isinstance(p,DTensor) else value.to(device))
        model.eval();torch.cuda.empty_cache()
        vae=AutoencoderKLMiniMaxH3.from_pretrained(args.base_source,subfolder='vae').to(device).eval()
        audio_vae=AutoencoderKLMiniMaxH3Audio.from_pretrained(args.base_source,subfolder='audio_vae').to(device).eval()
        decoder=FixedChunkVideoDecoder(vae,device)
        decoder_offload_during_dit=inference_reshard
        if decoder_offload_during_dit:
            vae.to('cpu');audio_vae.to('cpu');torch.cuda.empty_cache()
        load_seconds=time.monotonic()-start_load
        h,w=48,84;video=torch.zeros(24,source['video_latent_count'],h,w,dtype=torch.bfloat16)
        audio=torch.zeros(2,32,source['audio_latent_count'],dtype=torch.bfloat16)
        # Shape/validity only; no video.pt or audio.pt is read by this sampler.
        mask=torch.ones(source['audio_latent_count'],dtype=torch.bool)
        generator=torch.Generator(device=device).manual_seed(int(case.get('seed',20261014+rank//2)))
        generated_ids=set();active_history=GeneratedVideoHistory(source)
        pixels=[];audio_pieces=[];records=[];parent=None
        torch.cuda.reset_peak_memory_stats();dist.barrier()
        with torch.inference_mode():
          for number,row in enumerate(case['chunks']):
            chunk_start=time.monotonic();prompt=torch.load(row['prompt_path'],map_location='cpu',weights_only=True)
            if mode=='reset':active_history.reset()
            if args.adapter_mode=='native_stock':
                from src.inference.render import generate_latents
                chosen=[];peer_ids=[];noise_hash=None
                native_noise={}
                def capture_noise(v,a):
                    native_noise['video']=v.detach().cpu().contiguous()
                    native_noise['audio']=a.detach().cpu().contiguous()
                a,b=row['video_latent_start'],row['video_latent_end']
                native_frames=5+17*row['demo_units'];denoise_start=time.monotonic();steps=[]
                stock_video,stock_audio=generate_latents(model,prompt['prompt_embeds'].to(device),prompt['text_token_tags'].cpu(),
                    native_frames,args.nfe,int(case['seed'])+number,device,step_seconds=steps,initial_noise_callback=capture_noise)
                denoise_seconds=time.monotonic()-denoise_start
                initial_video_noise_cpu=native_noise['video'];initial_audio_noise_cpu=native_noise['audio']
                video_noise_sha256=hashlib.sha256(initial_video_noise_cpu.numpy().tobytes()).hexdigest()
                audio_noise_sha256=hashlib.sha256(initial_audio_noise_cpu.numpy().tobytes()).hexdigest()
                # Stock native video uses its own standalone 2-latent startup.
                stock_trim_start=0 if number==0 else 2
                chunk=stock_video[:,:,stock_trim_start:];assert chunk.shape[2]==b-a
                aa,ae=row['audio_latent_start'],row['audio_latent_end']
                audio_offset=0 if number==0 else round(5*40/24)
                gen_audio=stock_audio[:,:,audio_offset:audio_offset+ae-aa].to(torch.bfloat16)
                if gen_audio.shape[2]<ae-aa:gen_audio=torch.nn.functional.pad(gen_audio,(0,ae-aa-gen_audio.shape[2]))
                print(json.dumps(dict(event='native_shot_NFE_complete',rank=rank,shot=number,denoise_seconds=denoise_seconds)),flush=True)
            else:
                chosen,history_video=active_history.select(row['video_latent_start'])
                peer_ids=[]
                if mode=='wrong' and chosen:
                    history_video,peer_ids=wrong_past_video(peer['video'],peer_case['chunks'],number,len(chosen))
                batch=fixed_matched_batch(row,source,video,audio,prompt['prompt_embeds'],prompt['text_token_tags'],mask,
                    generator,device,retained_ids=chosen,retained_video=history_video)
                batch.update(sampling=True,zero_history_state=mode=='zero')
                prefix_reads_current=args.empty_history_attention=='full_document' and batch['history_rows'].shape[1]==0
                for processor in processors:
                    processor.prefix_rows=batch['prefix_rows'];processor.prefix_reads_current=prefix_reads_current
                    processor.text_rows=int(batch['inputs']['text_indices'].numel())
                inputs=batch['inputs'];noise_hash=hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest()
                xvideo=torch.randn(inputs['hidden_states'].shape,generator=generator,device=device)
                xaudio=torch.randn(inputs['audio_hidden_states'].shape,generator=generator,device=device)
                initial_video_noise_cpu=xvideo.detach().cpu().contiguous()
                initial_audio_noise_cpu=xaudio.detach().cpu().contiguous()
                video_noise_sha256=hashlib.sha256(initial_video_noise_cpu.numpy().tobytes()).hexdigest()
                audio_noise_sha256=hashlib.sha256(initial_audio_noise_cpu.numpy().tobytes()).hexdigest()
                if str(number) in expected_noise:
                    e=expected_noise[str(number)]
                    assert (noise_hash,video_noise_sha256,audio_noise_sha256)==(e['noise_state_sha256'],e['video_noise_sha256'],e['audio_noise_sha256'])
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
                    if i==0:
                        torch.cuda.synchronize()
                        if not torch.isfinite(pv).all() or not torch.isfinite(pa).all():raise RuntimeError('Nonfinite first model forward')
                        print(json.dumps(dict(event='first_NFE_complete',rank=rank,case_id=case['case_id'],chunk=number,world_size=world,
                            current_video_rows=int(xvideo.shape[1]),history_rows=int(batch['history_rows'].shape[1]),
                            first_forward_with_initial_gathers_seconds=time.monotonic()-denoise_start,
                            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)),flush=True)
                    xvideo=xvideo+(tvnext-tv)*pv.float();xaudio=xaudio+(tanext-ta)*pa.float()
                torch.cuda.synchronize();denoise_seconds=time.monotonic()-denoise_start
                if not torch.isfinite(xvideo).all() or not torch.isfinite(xaudio).all():raise RuntimeError('Nonfinite generated chunk')
            a,b=row['video_latent_start'],row['video_latent_end'];aa,ae=row['audio_latent_start'],row['audio_latent_end']
            if args.adapter_mode!='native_stock':chunk=unpack_video(xvideo,b-a,h,w).to(torch.bfloat16)
            chunk=chunk.to(torch.bfloat16)
            video[:,a:b]=chunk[0].cpu();generated_ids.update(range(a,b))
            active_history.put(a,chunk[0]);evicted=active_history.evict(b)
            if args.adapter_mode!='native_stock':gen_audio=xaudio[0].reshape(2,ae-aa,32).permute(0,2,1).to(torch.bfloat16)
            audio_pieces.append(gen_audio);audio[:,:,aa:ae]=gen_audio.cpu()
            snapshot_start=time.monotonic()
            snapshot=out/f"{case['case_id']}_{mode}_seed{case['seed']}_chunk{number:03d}_GENERATED.pt"
            temporary=snapshot.with_suffix('.pt.tmp')
            torch.save(dict(case_id=case['case_id'],adapter_mode=args.adapter_mode,state_mode=mode,nfe=args.nfe,seed=case['seed'],completed_NN_chunk=number,
                video=video,audio=audio,generated_ids=sorted(generated_ids),noise_generator_state=generator.get_state().cpu(),
                previous_complete_records=records,current_video_noise_sha256=video_noise_sha256,current_audio_noise_sha256=audio_noise_sha256,
                current_prompt_sha256=hashlib.sha256(Path(row['prompt_path']).read_bytes()).hexdigest(),current_chunk=row,initial_video_noise=initial_video_noise_cpu,initial_audio_noise=initial_audio_noise_cpu),temporary)
            os.replace(temporary,snapshot);snapshot_seconds=time.monotonic()-snapshot_start
            decode_start=time.monotonic();video_vae_h2d_seconds=0.;video_vae_d2h_seconds=0.
            if decoder_offload_during_dit:
                move_start=time.monotonic();vae.to(device);torch.cuda.synchronize();video_vae_h2d_seconds=time.monotonic()-move_start
            shot_pixel_start=len(pixels)
            cut=parent is not None and (parent!=row['parent_window_id'] or previous_end!=a)
            if cut:
                tail=decoder.flush()
                if tail is not None:pixels.append(tail.cpu())
            if args.adapter_mode=='native_stock':
                for part in decoder_chunks(stock_video,0):pixels.append(decoder.push(part).cpu())
                tail=decoder.flush()
                if tail is not None:pixels.append(tail.cpu())
                if number>0:
                    shot_pixels=torch.cat(pixels[shot_pixel_start:],dim=2)[:,:,5:]
                    pixels[shot_pixel_start:]=[shot_pixels]
            else:
                for part in decoder_chunks(chunk,a):pixels.append(decoder.push(part).cpu())
            torch.cuda.synchronize();decode_seconds=time.monotonic()-decode_start
            if decoder_offload_during_dit:
                move_start=time.monotonic();vae.to('cpu');torch.cuda.empty_cache();video_vae_d2h_seconds=time.monotonic()-move_start
            parent=row['parent_window_id'];previous_end=b
            record=dict(rank=rank,chunk=number,window_id=row['window_id'],case_id=case['case_id'],state_mode=mode,
                history_frame_ids=chosen,history_source=('none_native_independent_shots' if args.adapter_mode=='native_stock' else 'own_generated_video_only'),clean_gt_history_used=False,
                wrong_history_source=peer_case['source_id'] if peer_ids else None,wrong_history_frame_ids=peer_ids,
                active_history_bytes=active_history.bytes,evicted_frame_ids=evicted,
                sampling_protocol='user_prompt_program',generated_target_latents=b-a,
                empty_history_attention=args.empty_history_attention,text_read_policy=args.text_read_policy,text_prefix_reads_current=(False if args.adapter_mode=='native_stock' else (args.text_read_policy=='text_full' or prefix_reads_current)),
                current_prompt_sha256=hashlib.sha256(Path(row['prompt_path']).read_bytes()).hexdigest(),
                noise_state_sha256=noise_hash,video_noise_sha256=video_noise_sha256,audio_noise_sha256=audio_noise_sha256,source_target_start_sec=row['source_target_start_sec'],source_target_end_sec=row['source_target_end_sec'],
                hard_cut_decoder_reset=cut,model_history_reset=mode=='reset',nfe=args.nfe,cfg_scale=1.,
                denoise_seconds=denoise_seconds,video_decode_seconds=decode_seconds,chunk_wall_seconds=time.monotonic()-chunk_start,chunk_snapshot_seconds=snapshot_seconds,chunk_snapshot=str(snapshot),video_vae_h2d_seconds=video_vae_h2d_seconds,video_vae_d2h_seconds=video_vae_d2h_seconds)
            records.append(record);print(json.dumps(record),flush=True)
          tail=decoder.flush()
          if tail is not None:pixels.append(tail.cpu())
          decoded=torch.cat(pixels,dim=2)
          if args.sampling_protocol=='full_annotation':
            assert decoded.shape[2]==source['native_output_frames']
            decoded=decoded[:,:,:source['valid_output_frames']]

          path=out/f"{args.label}_{case['case_id']}_{mode}_seed{case['seed']}.mp4";encoding_start=time.monotonic()
          torch.save(dict(video=video,audio=audio,generated_ids=sorted(generated_ids),case_id=case['case_id']),out/f'rank{rank}_generated_latents.pt')
          artifact=video_file(path,decoded,case['nominal_output_fps'])
          # Audio is an offline complete-sequence decode, not a claimed online codec.
          if decoder_offload_during_dit:audio_vae.to(device)
          al=torch.cat(audio_pieces,dim=2);mean=torch.tensor(audio_vae.config.latents_mean,device=device).view(1,-1,1)
          std=torch.tensor(audio_vae.config.latents_std,device=device).view(1,-1,1)
          aw=audio_vae.decode(al*std+mean,return_dict=False)[0].float().permute(1,0,2)[0].clamp(-1,1).cpu()
          wav=path.with_suffix('.wav');pcm=(aw.T*32767).round().short().contiguous().numpy().tobytes()
          with wave.open(str(wav),'wb') as f:f.setnchannels(2);f.setsampwidth(2);f.setframerate(32000);f.writeframes(pcm)
          mux=path.with_name(path.stem+'.mux.mp4')
          subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-nostdin','-y','-i',str(path),'-i',str(wav),
              '-c:v','copy','-c:a','aac','-af','apad','-t',str(decoded.shape[2]/case['nominal_output_fps']),str(mux)],check=True)
          os.replace(mux,path)
          artifact.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),audio_muxed=True)
        result=dict(text_read_policy=args.text_read_policy,empty_history_attention=args.empty_history_attention,adapter_mode=args.adapter_mode,checkpoint_adapters_applied=args.adapter_mode=='trained',adapter_initialization='base proj_in copy + LoRA B=0' if args.adapter_mode=='native_init' else None,
            hardware_host=__import__('socket').gethostname(),hardware_gpu=torch.cuda.get_device_name(device),torch_version=torch.__version__,
            comparison_track='same_backbone_h3_engineering_diagnostic',inference_world_size=world,decoder_offload_during_dit=decoder_offload_during_dit,resolved_config_path=str(resolved_config_path),collective_timeout_min=args.collective_timeout_min,
            audio_future_context_allowed=False,audio_encoder_lookahead_sec=0.,source_audio_encoder_lookahead_sec=None,
            audio_valid_mask_after_encoder='N/A: no source audio encode',formal_av_alignment_status='missing',
            result_kind='user prompt program on explicit 24fps generated clock; native_stock has no cross-shot conditioning',source_data_audit_pass=None,generation_demo_only=True,source_pts=source.get('source_pts'),padding_output_frames=source.get('padding_output_frames'),native_output_frames=source.get('native_output_frames'),valid_output_frames=source.get('valid_output_frames'),output_timestamp_contract=source.get('output_timestamps'),status='prompt_program_demo_generated',rank=rank,case_id=case['case_id'],source_id=case['source_id'],
            checkpoint=(None if args.adapter_mode=='native_stock' else args.checkpoint),checkpoint_step=saved['step'],checkpoint_sha256=(None if args.adapter_mode=='native_stock' else hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()),
            state_mode=mode,comparison_mode=args.comparison_mode,sampling_protocol=args.sampling_protocol,
            generation_chunk_count=len(records),records=records,artifact=artifact,audio_wav=str(wav),
            nfe=args.nfe,cfg_scale=1.,video_generation_and_native_stream_decode=True,program_sha256=case.get('program_sha256'),
            clean_gt_latents_read=False,event_eval_ready=False,formal_memory_quality_claim=False,
            load_seconds=load_seconds,inference_parameter_resharding=inference_reshard,encode_and_audio_decode_seconds=time.monotonic()-encoding_start,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,offline_audio_decode_only=True)
        (out/f'rank{rank}_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');dist.barrier()
        if rank==0:
            all_results=[json.loads((out/f'rank{r}_RESULT.json').read_text()) for r in range(world)]
            for c in range(len(cases)):
                pair=all_results[c*2:c*2+2]
                if args.adapter_mode!='native_stock':
                    assert [r['video_noise_sha256'] for r in pair[0]['records']]==[r['video_noise_sha256'] for r in pair[1]['records']]
                    assert [r['audio_noise_sha256'] for r in pair[0]['records']]==[r['audio_noise_sha256'] for r in pair[1]['records']]
            (out/'GENERATION_PASS.json').write_text(json.dumps(dict(status=f'{world}_gpu_source_time_annotation_generated_history_interface_pass',
                videos=world,independent_sources=len(set(r['source_id'] for r in all_results)),source_seed_units=len(cases),comparison_mode=args.comparison_mode,generated_history_only=True,
                sampling_protocol=args.sampling_protocol,generation_chunks_per_video=len(case['chunks']),
                active_history_physically_bounded=True,matched_current_noise_per_pair=args.adapter_mode!='native_stock',event_eval_ready=False),indent=2)+'\n')
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
