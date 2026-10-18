"""InferHub-only source-clock execution validation; no formal quality admission."""
import argparse
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import time
import wave

import torch
import torch.distributed as dist

from train_h3_v2_local_history_ddp import bootstrap
bootstrap()
from h3_shared_loader import load_session
from h3_source_time_contract import load_case,SourceHistory,source_batch
from h3_source_frame_writer import SourceFrameWriter,SourceWindowTrim
from h3_fixed_stream_decode import FixedChunkVideoDecoder
from h3_native_shot_batch_infer import file_sha,tensor_sha
from h3_source_resume import save_progress,restore
from h3_source_state_probe import EffectiveHistoryProbe
from sample_demo import unpack_video


def validate(manifest):
    if manifest.get('policy')!='source_time_native_padding_trim_v1' or manifest.get('NFE')!=50:
        raise ValueError('Declare source-clock execution validation and 50 NFE')
    for pair in manifest['rounds']:
        programs=[load_case(t['case_file'])[1] for t in pair]
        if len(pair)!=2 or len({p['program_sha256'] for p in programs})!=1 or {t['history_mode'] for t in pair}!={'carry','zero'}:
            raise ValueError('Pair the same source/seed with carry and zero')
        remaining=[]
        for task,program in zip(pair,programs):
            count=task.get('resume_completed_chunks',0)
            if type(count) is not int or not 0<=count<=len(program['generation_chunks']):raise ValueError('Invalid source resume cursor')
            if count:
                prefix=task.get('resume_prefix_noise',[])
                if (not task.get('resume_archive') or len(prefix)!=count or file_sha(task['resume_archive'])!=task.get('resume_sha256') or
                    any(file_sha(p['path'])!=p['sha256'] for p in prefix)):
                    raise ValueError('Frozen source resume archive/input evidence changed')
            elif task.get('resume_archive'):raise ValueError('Source resume needs a nonzero declared committed cursor')
            remaining.append(len(program['generation_chunks'])-count)
        if len(set(remaining))!=1:raise ValueError('Paired source ranks require equal remaining model-forward counts')
        for program in programs:
            for clip in program['segments']:
                value=torch.load(clip['prompt_path'],map_location='cpu',weights_only=True)
                if (value['prompt']!=clip['prompt'] or value['prompt_embeds'].ndim!=2 or
                    value['prompt_embeds'].shape[1]!=5120 or not torch.isfinite(value['prompt_embeds']).all() or
                    value['text_token_tags'].shape!=(value['prompt_embeds'].shape[0],) or not bool((value['text_token_tags']==1).all())):
                    raise ValueError('Invalid source-window plaintext embedding')
        output=Path(manifest['output_root'])
        if not output.is_absolute() or not all(Path(t['output']).is_relative_to(output) for t in pair):
            raise ValueError('Source outputs must stay inside the declared job output root')


def render(session,task,spec):
    rank=dist.get_rank();device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    case,program=load_case(task['case_file']);rows=program['generation_chunks']
    out=Path(task['output']);out.mkdir(parents=True,exist_ok=True)
    if (out/f'rank{rank}_RESULT.json').exists():raise FileExistsError('Reuse completed source result instead of overwriting it')
    fingerprint=dict(torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(device),gpu_memory_bytes=torch.cuda.get_device_properties(device).total_memory,
        world_size=dist.get_world_size(),parameter_resharding=session['reshard'],
        sampler_sha256=file_sha(__file__),source_contract_sha256=file_sha(Path(__file__).with_name('h3_source_time_contract.py')),
        shared_loader_sha256=file_sha(Path(__file__).with_name('h3_shared_loader.py')))
    fingerprint['neural_code_sha256']={name:file_sha(importlib.import_module(name).__file__) for name in
        ['h3_spatial_history_flow','h3_prompt_history_attention','h3_fixed_chunk_contract','h3_fsdp2_speed',
         'diffusers.models.transformers.transformer_minimax_h3','diffusers.modular_pipelines.minimax_h3.before_denoise']}
    fingerprint['history_probe_sha256']=file_sha(Path(__file__).with_name('h3_source_state_probe.py'))
    fingerprint.update(cudnn_version=torch.backends.cudnn.version(),allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        float32_matmul_precision=torch.get_float32_matmul_precision())
    resume=None
    if task.get('resume_archive'):
        resume=dict(archive_path=task['resume_archive'],archive_sha256=task['resume_sha256'],
            completed_chunks=task['resume_completed_chunks'],prefix_noise=task['resume_prefix_noise'])
    config=out/f'rank{rank}_RESOLVED_CONFIG.json'
    config.write_text(json.dumps(dict(weights_spec=spec,case=case,NFE=50,CFG=1.,history_mode=task['history_mode'],
        source_time_contract='annotation_measured_pts_native_padding_trim_v1',formal_source_time_admitted=False,
        source_video_or_audio_condition=False,source_reference_encoder_evidence='not supplied by sampler',
        audio_history_used=False,audio_future_context_allowed=False,audio_encoder_lookahead_sec=0.,
        audio_valid_mask_after_encoder=None,audio_encoder_status='not used; generation from plaintext only',
        execution_fingerprint=fingerprint,source_resume=resume,effective_history_probe_required=True),indent=2)+'\n')
    model=session['model'];history=SourceHistory(program);generator=torch.Generator(device=device).manual_seed(program['seed'])
    records=[];generated={};first=0;replay_anchor=None
    identity=dict(program=program,case_sha256=file_sha(task['case_file']),weights_spec_sha256=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest(),
        checkpoint_sha256=spec.get('checkpoint_sha256'),mode=task['history_mode'],execution_fingerprint=fingerprint)
    if resume:
        generated,records,first=restore(resume['archive_path'],resume['archive_sha256'],**identity,history=history,generator=generator,
            set_cuda_rng_state=lambda state:torch.cuda.set_rng_state(state,device))
        if first!=resume['completed_chunks']:raise ValueError('Restored source prefix cursor changed')
        for record,proof in zip(records,resume['prefix_noise']):
            if record['noise_and_positions_path']!=proof['path'] or record['noise_and_positions_sha256']!=proof['sha256']:
                raise ValueError('Source resumed input provenance differs from the frozen prefix')
    torch.cuda.reset_peak_memory_stats();dist.barrier();started=time.monotonic()
    with torch.inference_mode():
        for number,row in enumerate(rows):
            if number<first:continue
            ci=row['clip_index'];clip=program['segments'][ci]
            prompt=torch.load(clip['prompt_path'],map_location='cpu',weights_only=True)
            batch=source_batch(row,program,prompt['prompt_embeds'],prompt['text_token_tags'],generator,device,history,zero_history=task['history_mode']=='zero')
            inputs=batch['inputs']
            for processor in session['processors']:
                processor.text_policy='text_full';processor.prefix_rows=batch['prefix_rows']
                processor.text_rows=inputs['text_indices'].numel();processor.prefix_reads_current=not row['history_frame_ids']
            xv=torch.randn(inputs['hidden_states'].shape,generator=generator,device=device)
            xa=torch.randn(inputs['audio_hidden_states'].shape,generator=generator,device=device)
            initial_v,initial_a=xv.detach().cpu().contiguous(),xa.detach().cpu().contiguous()
            positions={k:inputs['position_ids'][inputs[k+'_indices']].cpu() for k in ['text','state','audio','video']}
            noise=out/f'rank{rank}_chunk{number:03d}_NOISE_AND_POSITIONS.pt'
            torch.save(dict(initial_video_noise=initial_v,initial_audio_noise=initial_a,positions=positions,row=row),noise)
            torch.cuda.synchronize();tick=time.monotonic()
            state_probe=EffectiveHistoryProbe(model.transformer,task['history_mode'],batch['history_rows'].shape[1])
            try:
                for i in range(50):
                    u,unext=1-i/50.,1-(i+1)/50.
                    tv,ta=1-12*u/(1+11*u),1-3*u/(1+2*u)
                    vn,an=1-12*unext/(1+11*unext),1-3*unext/(1+2*unext)
                    times=torch.full((len(inputs['position_ids']),),tv,device=device)
                    times[inputs['state_indices']]=1.;times[inputs['audio_indices']]=ta
                    inputs['timestep'],inputs['timestep_indices']=torch.unique(times,sorted=True,return_inverse=True)
                    inputs['hidden_states'],inputs['audio_hidden_states']=xv,xa
                    pv,pa=model(batch);xv=xv+(vn-tv)*pv.float();xa=xa+(an-ta)*pa.float()
            finally:state_probe.close()
            observed_history=state_probe.summary()
            torch.cuda.synchronize();dit=time.monotonic()-tick
            a,b=row['clip_video_latent_start'],row['clip_video_latent_end'];aa,ae=row['audio_latent_start'],row['audio_latent_end']
            v=unpack_video(xv,b-a,48,84)[0].to(torch.bfloat16).cpu()
            audio=xa[0].reshape(2,ae-aa,32).permute(0,2,1).to(torch.bfloat16).cpu()
            if not torch.isfinite(v).all() or not torch.isfinite(audio).all():raise ValueError('Nonfinite generated source chunk')
            if ci not in generated:
                generated[ci]=dict(video=torch.zeros(24,clip['video_latent_count'],48,84,dtype=torch.bfloat16),
                    audio=torch.zeros(2,32,clip['audio_latent_count'],dtype=torch.bfloat16))
            generated[ci]['video'][:,a:b]=v;generated[ci]['audio'][:,:,aa:ae]=audio
            history_sha=tensor_sha(batch['history_rows']);evicted=history.commit(row,v)
            records.append(dict(chunk=number,row=row,NFE=50,CFG=1.,state_mode=task['history_mode'],denoise_seconds=dit,effective_history_input=observed_history,
                source_time_window=[row['source_target_start_sec'],row['model_target_end_sec']],
                history_rows_sha256=history_sha,history_state_bytes=history.bytes,evicted_parts=evicted,
                history_write_eligible=row['history_write_eligible'],video_noise_sha256=tensor_sha(initial_v),
                audio_noise_sha256=tensor_sha(initial_a),noise_and_positions_path=str(noise),noise_and_positions_sha256=file_sha(noise)))
            save_progress(out/f'rank{rank}_PROGRESS.pt',**identity,generated=generated,records=records,generator=generator,
                cuda_rng_state=torch.cuda.get_rng_state(device))
            # Retain one immutable real prefix before the episode ends. A later
            # GPU replay can exercise a genuine suffix, rather than reload a
            # completed tensor and incorrectly call that resume equivalence.
            if replay_anchor is None and number<len(rows)-1 and (evicted or number==len(rows)-2):
                anchor=out/f'rank{rank}_PROGRESS_REPLAY_ANCHOR.pt'
                if anchor.exists():raise FileExistsError('Do not overwrite an existing source replay anchor')
                save_progress(anchor,**identity,generated=generated,records=records,generator=generator,
                    cuda_rng_state=torch.cuda.get_rng_state(device))
                replay_anchor=dict(path=str(anchor),sha256=file_sha(anchor),completed_chunks=number+1,
                    remaining_chunks=len(rows)-number-1,after_actual_history_eviction=bool(evicted))
            print(json.dumps(dict(event='source_time_chunk_complete',chunk=number,rank=rank,seconds=dit)),flush=True)
        archive=out/f'rank{rank}_generated_latents.pt';torch.save(dict(generated_clips=generated,records=records),archive)
        decode_started=time.monotonic();vae=session['vae'];vae.to(device);path=out/(task['task_id']+'.mp4')
        origin,end=program['source_time_window']
        with SourceFrameWriter(path,program['actual_source_pts'],origin,end) as writer:
            for ci,clip in enumerate(program['segments']):
                decoder=FixedChunkVideoDecoder(vae,device);trim=SourceWindowTrim(clip['valid_output_frames'],clip['native_output_frames'])
                for row in [r for r in rows if r['clip_index']==ci]:
                    a,b=row['clip_video_latent_start'],row['clip_video_latent_end']
                    writer.append(trim.push(decoder.push(generated[ci]['video'][None,:,a:b].to(device))).cpu())
                tail=trim.push(decoder.flush())
                if tail is not None:writer.append(tail.cpu())
                trim.finish()
            artifact=writer.artifact()
        vae.cpu();torch.cuda.empty_cache();avae=session['audio_vae'];avae.to(device);waves=[];audio_records=[]
        mean=torch.tensor(avae.config.latents_mean,device=device).view(1,-1,1);std=torch.tensor(avae.config.latents_std,device=device).view(1,-1,1)
        for ci,clip in enumerate(program['segments']):
            audio=avae.decode(generated[ci]['audio'].to(device)*std+mean,return_dict=False)[0].float().permute(1,0,2)[0].clamp(-1,1).cpu()
            sa=round((clip['source_start_sec']-origin)*32000);sb=round((clip['source_end_sec']-origin)*32000);desired=sb-sa;actual=audio.shape[-1]
            audio_records.append(dict(clip_index=ci,start_sample=sa,end_sample=sb,decoded_samples=actual,
                trimmed_samples=max(0,actual-desired),padded_samples=max(0,desired-actual),
                valid_generated_sample_interval=[sa,sa+min(actual,desired)]))
            if actual<desired:audio=torch.nn.functional.pad(audio,(0,desired-actual))
            waves.append(audio[:,:desired])
        pcm=torch.cat(waves,dim=-1);wav=path.with_suffix('.wav')
        with wave.open(str(wav),'wb') as stream:
            stream.setnchannels(2);stream.setsampwidth(2);stream.setframerate(32000)
            stream.writeframes((pcm.T*32767).round().short().contiguous().numpy().tobytes())
        import imageio_ffmpeg
        mux=path.with_suffix('.mux.mp4')
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-nostdin','-y','-copyts','-i',str(path),'-i',str(wav),
            '-c:v','copy','-c:a','aac','-video_track_timescale','1000000','-avoid_negative_ts','disabled','-t',str(end-origin),str(mux)],check=True,timeout=120)
        import av
        with av.open(str(mux)) as video:
            pts=[float(f.pts*f.time_base)+origin for f in video.decode(video=0)]
        if len(pts)!=len(program['actual_source_pts']) or max(abs(a-b) for a,b in zip(pts,program['actual_source_pts']))>.00000051:
            raise ValueError('Audio mux shifted source video timestamps')
        mux.replace(path);artifact.update(sha256=file_sha(path),audio_muxed=True,post_mux_video_pts_verified=True)
        avae.cpu();torch.cuda.empty_cache()
    result=dict(status='source_clock_generated_unadmitted',artifact=artifact,records=records,resolved_config_path=str(config),
        program_sha256=program['program_sha256'],source_id=program['source_id'],checkpoint_sha256=spec.get('checkpoint_sha256'),
        checkpoint_step=session['checkpoint_step'],text_read_policy='text_full',weights_kind=session['weights_kind'],
        weights_load_scope='shared session; do not sum per-video load_seconds',audio_valid_mask_after_encoder=None,
        state_mode=task['history_mode'],comparison_track='same_backbone_h3_source_clock_execution_diagnostic',
        formal_source_time_admitted=False,formal_memory_quality_claim=False,event_eval_ready=False,
        source_pixels_or_audio_read=False,source_reference_audio_valid_mask_after_encoder=None,
        audio_future_context_allowed=False,audio_encoder_lookahead_sec=0.,audio_decode=audio_records,
        first_playable_av_chunk_seconds=None,source_sampler_resume_verified=False,
        source_resume=resume,restored_source_chunks=first,new_model_evaluations=50*(len(rows)-first),
        execution_fingerprint=fingerprint,timing_scope='continuation only; original prefix time is retained per chunk' if first else 'fresh complete source episode',
        timing_includes_first_NFE_history_probe=True,effective_history_probe_scope='first NFE per newly generated chunk',
        replay_validation_anchor=replay_anchor,timing_includes_validation_snapshot_io=True,
        load_seconds=session['load_seconds'],generation_and_export_seconds=time.monotonic()-started,
        decode_mux_seconds=time.monotonic()-decode_started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        hardware_gpu=torch.cuda.get_device_name(device),generated_archive=str(archive))
    temporary=out/f'rank{rank}_RESULT.tmp';temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(out/f'rank{rank}_RESULT.json')
    dist.barrier()


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--validate-inputs-only',action='store_true');a=p.parse_args()
    manifest=json.loads(Path(a.manifest).read_text());validate(manifest)
    if a.validate_inputs_only:print('SOURCE_INPUTS_VALIDATED_NO_MODEL');return
    if (not os.environ.get('INFER_CODE_DIR') or not os.environ.get('INFER_WEIGHTS_DIR') or
        os.environ.get('INFER_OUTPUT_DIR')!=manifest['output_root']):
        raise ValueError('Source neural execution requires the declared official InferHub worker environment')
    rank,world,local=[int(os.environ[k]) for k in ['RANK','WORLD_SIZE','LOCAL_RANK']]
    if world!=2:raise ValueError('Use two ranks per source-clock model session')
    torch.cuda.set_device(local);torch.set_num_threads(2);dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=60))
    try:
        session=load_session(manifest['weights_spec'],torch.device('cuda',local))
        for pair in manifest['rounds']:render(session,pair[rank],manifest['weights_spec'])
    except BaseException:
        import traceback
        try:traceback.print_exc()
        except OSError:pass
        os._exit(1)
    else:dist.destroy_process_group()


if __name__=='__main__':main()
