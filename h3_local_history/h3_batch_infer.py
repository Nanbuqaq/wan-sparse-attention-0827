"""Batched H3 sampler; model parameters load once, case histories reset per task.

The denoising and native decode loop is derived from sample_demo.py. New full
exports use the same current-text/history visibility as their training contract.
"""
import argparse,datetime,hashlib,json,os,subprocess,time,wave
from pathlib import Path
from types import SimpleNamespace
import imageio_ffmpeg
import torch
import torch.distributed as dist
from train_h3_v2_local_history_ddp import bootstrap
bootstrap()
from h3_shared_loader import load_session
from h3_fixed_chunk_contract import fixed_matched_batch
from h3_fixed_stream_decode import FixedChunkVideoDecoder
from h3_generated_video_history import GeneratedVideoHistory
from h3_variable_sampling_case import decoder_chunks
from sample_demo import unpack_video
from h3_frame_writer import FrameWriter


def render(session,task,spec,nfe):
    rank,world=dist.get_rank(),dist.get_world_size()
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    case=json.loads(Path(task['case_file']).read_text())['cases'][0]
    source=json.loads(Path(task['source_contracts']).read_text())[case['source_id']]
    mode=task['history_mode'];peer=None;peer_case=None
    if mode not in ('carry','zero','native'):raise ValueError('Unsupported history mode')
    out=Path(task['output']);out.mkdir(parents=True,exist_ok=True)
    args=SimpleNamespace(nfe=nfe,label=task['task_id'],adapter_mode=('native_stock' if mode=='native' else ('native_init' if spec['kind']=='base' else 'trained')),
        comparison_mode='zero',sampling_protocol='full_annotation',empty_history_attention='full_document',text_read_policy='text_full',
        collective_timeout_min=60,checkpoint=spec.get('checkpoint'))
    model,processors,vae,audio_vae=[session[x] for x in ['model','processors','vae','audio_vae']]
    decoder=FixedChunkVideoDecoder(vae,device)
    decoder_offload_during_dit=inference_reshard=session['reshard']
    saved={'step':session['checkpoint_step']};load_seconds=session['load_seconds']
    guards=task.get('expected_noise',{});expected_noise=guards
    resolved_config_path=out/f'rank{rank}_RESOLVED_CONFIG.json'
    resolved=dict(weights_spec=spec,sampler_args=vars(args),case=case,source_contract=source,
        actual_sampling_NFE=nfe,actual_CFG=1.,actual_valid_video_frames=source['valid_output_frames'],world_size=world,
        batch_session_reused=True,weights_load_seconds=load_seconds,
        GPU=torch.cuda.get_device_name(device),torch_version=torch.__version__)
    resolved_config_path.write_text(json.dumps(resolved,default=str,indent=2)+'\n')
    h,w=48,84;video=torch.zeros(24,source['video_latent_count'],h,w,dtype=torch.bfloat16)
    audio=torch.zeros(2,32,source['audio_latent_count'],dtype=torch.bfloat16)
    # Shape/validity only; no video.pt or audio.pt is read by this sampler.
    mask=torch.ones(source['audio_latent_count'],dtype=torch.bool)
    generator=torch.Generator(device=device).manual_seed(int(case.get('seed',20261014+rank//2)))
    generated_ids=set();active_history=GeneratedVideoHistory(source);native_noise_replay_source=None
    audio_pieces=[];records=[];parent=None
    path=out/f"{args.label}_{case['case_id']}_{mode}_seed{case['seed']}.mp4"
    torch.cuda.reset_peak_memory_stats();dist.barrier()
    with torch.inference_mode(), FrameWriter(path,case['nominal_output_fps']) as writer:
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
                if str(number) in expected_noise:
                    e=expected_noise[str(number)]
                    assert hashlib.sha256(native_noise['video'].numpy().tobytes()).hexdigest()==e['video_noise_sha256']
                    assert hashlib.sha256(native_noise['audio'].numpy().tobytes()).hexdigest()==e['audio_noise_sha256']
            a,b=row['video_latent_start'],row['video_latent_end']
            override=None
            if task.get('native_noise_replay_snapshot'):
                if number or a or len(case['chunks'])!=1:raise ValueError('Native noise replay requires one empty-history span')
                replay_path=Path(task['native_noise_replay_snapshot']);replay=torch.load(replay_path,map_location='cpu',weights_only=True,mmap=True)
                rv,ra=replay['initial_video_noise'],replay['initial_audio_noise']
                if rv.ndim not in (3,5) or ra.ndim not in (2,3):raise ValueError('Unknown saved native replay noise layout')
                override=dict(video=unpack_video(rv,b-a,h,w) if rv.ndim==3 else rv,audio=ra[0] if ra.ndim==3 else ra)
                native_noise_replay_source=dict(path=str(replay_path),sha256=hashlib.sha256(replay_path.read_bytes()).hexdigest(),
                    source_video_noise_shape=list(rv.shape),source_audio_noise_shape=list(ra.shape),
                    source_video_noise_sha256=hashlib.sha256(rv.contiguous().numpy().tobytes()).hexdigest(),
                    source_audio_noise_sha256=hashlib.sha256(replay['initial_audio_noise'].contiguous().numpy().tobytes()).hexdigest())
            native_frames=5+17*row['demo_units'];denoise_start=time.monotonic();steps=[]
            stock_video,stock_audio=generate_latents(model,prompt['prompt_embeds'].to(device),prompt['text_token_tags'].cpu(),
                native_frames,args.nfe,int(case['seed'])+number,device,step_seconds=steps,initial_noise_callback=capture_noise,initial_noise_override=override)
            if len(steps)!=args.nfe:raise ValueError('Native sampler model-evaluation count differs')
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
        if args.adapter_mode=='native_stock' and number>0:writer.skip(5)
        cut=parent is not None and (parent!=row['parent_window_id'] or previous_end!=a)
        if cut:
            tail=decoder.flush()
            if tail is not None:writer.append(tail.cpu())
        if args.adapter_mode=='native_stock':
            for part in decoder_chunks(stock_video,0):writer.append(decoder.push(part).cpu())
            tail=decoder.flush()
            if tail is not None:writer.append(tail.cpu())
        else:
            for part in decoder_chunks(chunk,a):writer.append(decoder.push(part).cpu())
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
      if tail is not None:writer.append(tail.cpu())
      writer.close();frame_count=writer.frames
      if args.sampling_protocol=='full_annotation':
        assert frame_count==source['native_output_frames']==source['valid_output_frames']

      encoding_start=time.monotonic()
      torch.save(dict(video=video,audio=audio,generated_ids=sorted(generated_ids),case_id=case['case_id']),out/f'rank{rank}_generated_latents.pt')
      artifact=writer.artifact()
      # Audio is an offline complete-sequence decode, not a claimed online codec.
      if decoder_offload_during_dit:audio_vae.to(device)
      al=torch.cat(audio_pieces,dim=2);mean=torch.tensor(audio_vae.config.latents_mean,device=device).view(1,-1,1)
      std=torch.tensor(audio_vae.config.latents_std,device=device).view(1,-1,1)
      aw=audio_vae.decode(al*std+mean,return_dict=False)[0].float().permute(1,0,2)[0].clamp(-1,1).cpu()
      wav=path.with_suffix('.wav');pcm=(aw.T*32767).round().short().contiguous().numpy().tobytes()
      with wave.open(str(wav),'wb') as f:f.setnchannels(2);f.setsampwidth(2);f.setframerate(32000);f.writeframes(pcm)
      mux=path.with_name(path.stem+'.mux.mp4')
      subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-nostdin','-y','-i',str(path),'-i',str(wav),
          '-c:v','copy','-c:a','aac','-af','apad','-t',str(frame_count/case['nominal_output_fps']),str(mux)],check=True,timeout=120)
      os.replace(mux,path)
      artifact.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),audio_muxed=True)
    result=dict(text_read_policy=args.text_read_policy,empty_history_attention=args.empty_history_attention,adapter_mode=args.adapter_mode,checkpoint_adapters_applied=session['weights_kind']=='adapter',full_model_weights_applied=session['weights_kind']=='full',weights_kind=session['weights_kind'],training_metadata=session['training_metadata'],adapter_initialization='base proj_in copy + LoRA B=0' if args.adapter_mode=='native_init' else None,
        hardware_host=__import__('socket').gethostname(),hardware_gpu=torch.cuda.get_device_name(device),torch_version=torch.__version__,
        comparison_track='same_backbone_h3_engineering_diagnostic',inference_world_size=world,decoder_offload_during_dit=decoder_offload_during_dit,resolved_config_path=str(resolved_config_path),collective_timeout_min=args.collective_timeout_min,
        audio_future_context_allowed=False,audio_encoder_lookahead_sec=0.,source_audio_encoder_lookahead_sec=None,
        audio_valid_mask_after_encoder='N/A: no source audio encode',formal_av_alignment_status='missing',
        result_kind='user prompt program on explicit 24fps generated clock; native_stock has no cross-shot conditioning',source_data_audit_pass=None,generation_demo_only=True,source_pts=source.get('source_pts'),padding_output_frames=source.get('padding_output_frames'),native_output_frames=source.get('native_output_frames'),valid_output_frames=source.get('valid_output_frames'),output_timestamp_contract=source.get('output_timestamps'),status='prompt_program_demo_generated',rank=rank,case_id=case['case_id'],source_id=case['source_id'],
        checkpoint=(None if args.adapter_mode=='native_stock' else args.checkpoint),checkpoint_step=saved['step'],checkpoint_sha256=(None if args.adapter_mode=='native_stock' else session['checkpoint_sha256']),
        state_mode=mode,comparison_mode=args.comparison_mode,sampling_protocol=args.sampling_protocol,
        generation_chunk_count=len(records),records=records,artifact=artifact,audio_wav=str(wav),
        nfe=args.nfe,cfg_scale=1.,video_generation_and_native_stream_decode=True,program_sha256=case.get('program_sha256'),
        clean_gt_latents_read=False,event_eval_ready=False,formal_memory_quality_claim=False,
        load_seconds=load_seconds,inference_parameter_resharding=inference_reshard,encode_and_audio_decode_seconds=time.monotonic()-encoding_start,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,offline_audio_decode_only=True)
    result.update(batch_session_reused=True,weights_load_scope='shared session, do not sum per-video load_seconds',chunk_policy=case.get('chunk_policy'))
    if native_noise_replay_source:result['native_noise_replay_source']=native_noise_replay_source
    (out/f'rank{rank}_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');dist.barrier()
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);args=ap.parse_args()
    manifest=json.loads(Path(args.manifest).read_text())
    rank,world,local=[int(os.environ[k]) for k in ['RANK','WORLD_SIZE','LOCAL_RANK']]
    torch.cuda.set_device(local);torch.set_num_threads(2)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=60))
    spec=manifest['weights_spec'];session=None;results=[]
    try:
        session=load_session(spec,torch.device('cuda',local),dense=manifest.get('dense',False))
        session_dir=Path(manifest['session_output']);session_dir.mkdir(parents=True,exist_ok=True)
        (session_dir/f'rank{rank}_LOAD.json').write_text(json.dumps({k:v for k,v in session.items() if k not in ['model','processors','vae','audio_vae','decoder']},default=str,indent=2)+'\n')
        for round_index,round_tasks in enumerate(manifest['rounds']):
            if len(round_tasks)!=world:raise ValueError('One task per rank is required in every round')
            shapes=[]
            for task in round_tasks:
                case=json.loads(Path(task['case_file']).read_text())['cases'][0]
                shapes.append(len(case['chunks']))
            if len(set(shapes))!=1:raise ValueError('Ranks must execute the same number of denoising chunks')
            torch.cuda.empty_cache()
            result=render(session,round_tasks[rank],spec,manifest.get('NFE',50));results.append(result)
            if rank==0:print(json.dumps(dict(event='batch_round_complete',round=round_index)),flush=True)
        (session_dir/f'rank{rank}_BATCH_COMPLETE.json').write_text(json.dumps(dict(tasks=len(results),results=[r['artifact'] for r in results]),indent=2)+'\n')
    finally:
        dist.destroy_process_group()

if __name__=='__main__':main()
