"""Persistent H3 sessions for native-shot cases; no teacher media is consumed."""
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

from train_h3_v2_local_history_ddp import bootstrap
bootstrap()
from h3_shared_loader import load_session
from h3_native_resume import save_progress,restore
from h3_native_shot_contract import load_case,native_rows,NativeHistory,native_batch
from h3_fixed_stream_decode import FixedChunkVideoDecoder
from h3_frame_writer import FrameWriter
from sample_demo import unpack_video


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def tensor_sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def validate(manifest):
    if manifest.get('policy')!='native_shot' or manifest.get('NFE')!=50:
        raise ValueError('Declare the native-shot 50-NFE inference protocol')
    for tasks in manifest['rounds']:
        counts=[]
        for task in tasks:
            case,episode=load_case(task['case_file'])
            if task['history_mode'] not in ['carry','zero']:
                raise ValueError('Unsupported history control')
            for path in {row['prompt_path'] for row in native_rows(episode)}:
                value=torch.load(path,map_location='cpu',weights_only=True)
                if (value['prompt_embeds'].ndim!=2 or value['prompt_embeds'].shape[1]!=5120 or
                    value['text_token_tags'].shape!=(value['prompt_embeds'].shape[0],) or
                    not bool((value['text_token_tags']==1).all()) or not torch.isfinite(value['prompt_embeds']).all()):
                    raise ValueError('Invalid plaintext embedding')
            counts.append(len(native_rows(episode))-int(task.get('resume_completed_chunks',0)))
        if len(tasks)!=2 or len(set(counts))!=1:
            raise ValueError('A two-rank round must have equal complete forward counts')


def render(session,task,spec):
    rank=dist.get_rank();device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    case,episode=load_case(task['case_file']);rows=native_rows(episode)
    out=Path(task['output']);out.mkdir(parents=True,exist_ok=True)
    if (out/f'rank{rank}_RESULT.json').exists():
        raise FileExistsError('Completed native case already exists; reuse it or choose a new output directory')
    mode=task['history_mode'];model=session['model'];processors=session['processors']
    resolved=out/f'rank{rank}_RESOLVED_CONFIG.json'
    resolved.write_text(json.dumps(dict(weights_spec=spec,case={**case, 'nominal_output_fps':24.},episode=episode,
        native_rows=rows,NFE=50,CFG=1.,history_mode=mode,model_load_scope='shared session',
        protocol='native_shot_startup7_fixed5',text_read_policy='text_full',
        source_pts=None,audio_encoder_lookahead_sec=0.,audio_future_context_allowed=False,
        audio_decode='per-shot offline decode; exact output duration recorded'),indent=2)+'\n')
    history=NativeHistory(episode);generator=torch.Generator(device=device).manual_seed(case['seed'])
    generated={};records=[];first=0
    if task.get('resume_archive'):
        generated,records,first=restore(task['resume_archive'],task['resume_sha256'],case_sha256=file_sha(task['case_file']),
            checkpoint_sha256=spec.get('checkpoint_sha256'),weights_spec_sha256=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest(),case_id=case['case_id'],mode=mode,rows=rows,episode=episode,history=history,generator=generator)
        if first!=task.get('resume_completed_chunks'):raise ValueError('Native resume count differs from scheduled peer counts')
    torch.cuda.reset_peak_memory_stats();dist.barrier();started=time.monotonic()
    with torch.inference_mode():
        for number,row in enumerate(rows):
            if number<first:continue
            ci=row['clip_index'];clip=episode['clips'][ci]
            prompt=torch.load(row['prompt_path'],map_location='cpu',weights_only=True)
            batch=native_batch(row,episode,prompt['prompt_embeds'],prompt['text_token_tags'],generator,device,history,zero_history=mode=='zero')
            inputs=batch['inputs']
            for processor in processors:
                processor.text_policy='text_full';processor.prefix_rows=batch['prefix_rows']
                processor.text_rows=inputs['text_indices'].numel();processor.prefix_reads_current=not row['history_frame_ids']
            noise_state=tensor_sha(generator.get_state());xv=torch.randn(inputs['hidden_states'].shape,generator=generator,device=device)
            xa=torch.randn(inputs['audio_hidden_states'].shape,generator=generator,device=device)
            initial_v,initial_a=xv.detach().cpu().contiguous(),xa.detach().cpu().contiguous()
            positions={k:inputs['position_ids'][inputs[k+'_indices']].float().cpu() for k in ['text','state','audio','video']}
            history_digest=tensor_sha(batch['history_rows']);torch.cuda.synchronize();tick=time.monotonic()
            for i in range(50):
                u,unext=1-i/50.,1-(i+1)/50.
                tv,ta=1-12*u/(1+11*u),1-3*u/(1+2*u)
                vn,an=1-12*unext/(1+11*unext),1-3*unext/(1+2*unext)
                times=torch.full((len(inputs['position_ids']),),tv,device=device)
                times[inputs['state_indices']]=1.;times[inputs['audio_indices']]=ta
                inputs['timestep'],inputs['timestep_indices']=torch.unique(times,sorted=True,return_inverse=True)
                inputs['hidden_states'],inputs['audio_hidden_states']=xv,xa
                pv,pa=model(batch);xv=xv+(vn-tv)*pv.float();xa=xa+(an-ta)*pa.float()
                if i in [0,49]:
                    torch.cuda.synchronize()
                    print(json.dumps(dict(event='native_shot_NFE',rank=rank,case=task['case_name'],mode=mode,chunk=number,NFE=i+1,seconds=time.monotonic()-tick)),flush=True)
            torch.cuda.synchronize();dit=time.monotonic()-tick
            a,b=row['clip_video_latent_start'],row['clip_video_latent_end'];aa,ae=row['audio_latent_start'],row['audio_latent_end']
            v=unpack_video(xv,b-a,48,84)[0].to(torch.bfloat16).cpu()
            audio=xa[0].reshape(2,ae-aa,32).permute(0,2,1).to(torch.bfloat16).cpu()
            if not torch.isfinite(v).all() or not torch.isfinite(audio).all():raise ValueError('Nonfinite generated native chunk')
            if ci not in generated:
                generated[ci]=dict(video=torch.zeros(24,clip['video_latent_count'],48,84,dtype=torch.bfloat16),
                                   audio=torch.zeros(2,32,round(clip['num_frames']/24.*40),dtype=torch.bfloat16))
            generated[ci]['video'][:,a:b]=v;generated[ci]['audio'][:,:,aa:ae]=audio
            evicted=history.commit(row,v);noise_file=out/f'rank{rank}_chunk{number:03d}_NOISE_AND_POSITIONS.pt'
            torch.save(dict(initial_video_noise=initial_v,initial_audio_noise=initial_a,positions_fp32=positions,row=row),noise_file)
            record=dict(chunk=number,window_id=row['window_id'],clip_id=clip['clip_id'],row=row,state_mode=mode,
                generated_target_latents=b-a,history_frame_ids=row['history_frame_ids'],history_parts=row['history_parts'],
                history_rows_sha256=history_digest,history_source='committed_generated_native_shot_chunks_only',
                active_history_bytes=history.bytes,evicted_frame_ids=evicted,
                current_prompt_sha256=file_sha(row['prompt_path']),noise_state_sha256=noise_state,
                video_noise_sha256=tensor_sha(initial_v),audio_noise_sha256=tensor_sha(initial_a),
                video_noise_shape=list(initial_v.shape),audio_noise_shape=list(initial_a.shape),
                positions_fp32_sha256={k:tensor_sha(x) for k,x in positions.items()},noise_and_positions_path=str(noise_file),
                nfe=50,cfg_scale=1.,denoise_seconds=dit,source_target_start_sec=row['source_target_start_sec'],source_target_end_sec=row['source_target_end_sec'],
                teacher_clean_media_read=False,student_reference_image_read=False)
            records.append(record)
            save_progress(out/f'rank{rank}_PROGRESS.pt',case_sha256=file_sha(task['case_file']),checkpoint_sha256=spec.get('checkpoint_sha256'),weights_spec_sha256=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest(),
                case_id=case['case_id'],mode=mode,generated=generated,records=records,generator=generator)
            print(json.dumps(dict(event='native_shot_chunk_complete',rank=rank,**record)),flush=True)
        archive=out/f'rank{rank}_generated_latents.pt';torch.save(dict(generated_clips=generated,records=records,case_id=case['case_id']),archive)
        vae=session['vae'];vae.to(device);path=out/(task['task_id']+'.mp4');decode_start=time.monotonic()
        with FrameWriter(path,24.) as writer:
            for ci,clip in enumerate(episode['clips']):
                decoder=FixedChunkVideoDecoder(vae,device)
                for row in [r for r in rows if r['clip_index']==ci]:
                    a,b=row['clip_video_latent_start'],row['clip_video_latent_end']
                    writer.append(decoder.push(generated[ci]['video'][None,:,a:b].to(device)).cpu())
                tail=decoder.flush()
                if tail is not None:writer.append(tail.cpu())
            writer.close();artifact=writer.artifact()
        expected=sum(c['num_frames'] for c in episode['clips'])
        if artifact['frames']!=expected:raise ValueError('Native-shot decode changed the requested frame count')
        vae.cpu();torch.cuda.empty_cache();avae=session['audio_vae'];avae.to(device);waves=[];audio_decode=[]
        mean=torch.tensor(avae.config.latents_mean,device=device).view(1,-1,1);std=torch.tensor(avae.config.latents_std,device=device).view(1,-1,1)
        for ci,clip in enumerate(episode['clips']):
            audio=avae.decode(generated[ci]['audio'].to(device)*std+mean,return_dict=False)[0].float().permute(1,0,2)[0].clamp(-1,1).cpu()
            desired=round(clip['num_frames']/24.*32000);actual=audio.shape[-1]
            audio_decode.append(dict(clip_id=clip['clip_id'],decoded_samples=actual,delivered_samples=desired,
                                     trimmed_samples=max(0,actual-desired),padded_samples=max(0,desired-actual)))
            if actual<desired:audio=torch.nn.functional.pad(audio,(0,desired-actual))
            waves.append(audio[:,:desired])
        pcm=torch.cat(waves,dim=-1);wav=path.with_suffix('.wav')
        with wave.open(str(wav),'wb') as stream:
            stream.setnchannels(2);stream.setsampwidth(2);stream.setframerate(32000)
            stream.writeframes((pcm.T*32767).round().short().contiguous().numpy().tobytes())
        mux=path.with_suffix('.mux.mp4')
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-nostdin','-y','-i',str(path),'-i',str(wav),
                        '-c:v','copy','-c:a','aac','-t',str(expected/24.),str(mux)],check=True,timeout=120)
        mux.replace(path);artifact.update(sha256=file_sha(path),audio_muxed=True)
        avae.cpu();torch.cuda.empty_cache()
    result=dict(status='native_shot_prompt_program_generated',artifact=artifact,records=records,resolved_config_path=str(resolved),
        checkpoint=spec.get('checkpoint'),checkpoint_sha256=spec.get('checkpoint_sha256'),checkpoint_step=session['checkpoint_step'],
        weights_kind=session['weights_kind'],training_metadata=session['training_metadata'],text_read_policy='text_full',state_mode=mode,
        nfe=50,cfg_scale=1.,chunk_policy='native_shot',native_boundary_policy='startup7_each_shot_then_regular5',
        comparison_track='same_backbone_h3_engineering_diagnostic',generated_clock_only=True,formal_memory_quality_claim=False,event_eval_ready=False,
        audio_future_context_allowed=False,audio_encoder_lookahead_sec=0.,audio_valid_mask_after_encoder='N/A: no source audio encoding',
        offline_native_shot_audio_decode_only=True,audio_decode=audio_decode,generated_archive=str(archive),
        teacher_clean_media_read=False,student_reference_image_read=False,complete_episode=True,
        load_seconds=session['load_seconds'],weights_load_scope='shared session, do not sum per-video load_seconds',
        generation_and_export_seconds=time.monotonic()-started,decode_mux_seconds=time.monotonic()-decode_start,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,hardware_host=__import__('socket').gethostname(),
        hardware_gpu=torch.cuda.get_device_name(device),hardware_total_gib=torch.cuda.get_device_properties(device).total_memory/2**30,
        torch_version=torch.__version__,actual_parameter_resharding=session['reshard'],restored_native_chunks=first,new_model_evaluations=50*(len(rows)-first),
        generated_prefix_checkpoint_policy='atomic complete generated-only prefix and RNG after every native chunk')
    temporary=out/f'rank{rank}_RESULT.tmp';temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(out/f'rank{rank}_RESULT.json')
    dist.barrier()
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--validate-inputs-only',action='store_true');args=parser.parse_args()
    manifest=json.loads(Path(args.manifest).read_text());validate(manifest)
    if args.validate_inputs_only:
        print('NATIVE_INPUTS_VALIDATED_NO_MODEL_FORWARD',len(manifest['rounds']));return
    rank,world,local=[int(os.environ[k]) for k in ['RANK','WORLD_SIZE','LOCAL_RANK']]
    if world!=2:raise ValueError('Use two ranks per persistent native-shot session')
    torch.cuda.set_device(local);torch.set_num_threads(2);dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=60))
    try:
        session=load_session(manifest['weights_spec'],torch.device('cuda',local))
        directory=Path(manifest['session_output']);directory.mkdir(parents=True,exist_ok=True)
        (directory/f'rank{rank}_LOAD.json').write_text(json.dumps({k:v for k,v in session.items() if k not in ['model','processors','vae','audio_vae','decoder']},default=str,indent=2)+'\n')
        for tasks in manifest['rounds']:render(session,tasks[rank],manifest['weights_spec'])
        (directory/f'rank{rank}_COMPLETE.json').write_text(json.dumps(dict(complete=True,tasks=len(manifest['rounds'])))+'\n')
    except BaseException:
        # Do not spend the NCCL timeout in destroy_process_group after one rank
        # fails. torchrun observes this exit and terminates its sibling promptly.
        import traceback
        try:traceback.print_exc()
        except OSError:pass
        os._exit(1)
    else:dist.destroy_process_group()


if __name__=='__main__':main()
