"""One native return chunk from a fixed prefix and actual saved target noise."""
import argparse,datetime,json,os,socket,subprocess,time,wave
from pathlib import Path
import imageio_ffmpeg
import torch
import torch.distributed as dist
from train_h3_v2_local_history_ddp import bootstrap
bootstrap()
from h3_shared_loader import load_session
from h3_native_shot_contract import native_batch
from h3_native_history_probe_contract import load_inputs,intervene_raw,tensor_sha,file_sha,POLICIES
from h3_history_content_contract import content_mask,mask_projected
from h3_fixed_stream_decode import FixedChunkVideoDecoder
from h3_frame_writer import FrameWriter
from sample_demo import unpack_video


def run(session,task,device):
    policy=task['policy']
    if policy not in POLICIES:raise ValueError('Unknown probe policy')
    root=Path(task['input'])
    if file_sha(root/'INPUT.json')!=task['input_sha256']:raise ValueError('Probe manifest/input binding changed')
    meta,case,episode,history,noise,donor=load_inputs(root)
    generation_started=time.monotonic()
    row=meta['target_row'];out=Path(task['output']);out.mkdir(parents=True,exist_ok=True)
    if (out/'RESULT.json').exists():raise FileExistsError('Reuse completed probe')
    prompt=torch.load(row['prompt_path'],map_location='cpu',weights_only=True)
    generator=torch.Generator(device=device).manual_seed(case['seed'])
    batch=native_batch(row,episode,prompt['prompt_embeds'],prompt['text_token_tags'],generator,device,history)
    inputs=batch['inputs'];ids=row['history_frame_ids']
    if tensor_sha(batch['history_rows'])!=meta['target_record']['history_rows_sha256']:
        raise ValueError('Replayed prefix content changed')
    positions={k:inputs['position_ids'][inputs[k+'_indices']].float().cpu() for k in ['text','state','audio','video']}
    hashes={k:tensor_sha(v) for k,v in positions.items()}
    if hashes!=meta['target_record']['positions_fp32_sha256']:raise ValueError('Actual positions changed')
    batch['history_rows']=intervene_raw(batch['history_rows'],ids,meta['original_ids'],donor['rows'],policy)
    mask=content_mask(ids,meta['original_ids'],meta['recent_ids'],policy if policy!='wrong_original' else 'full',1008,device)
    seen=dict(calls=0,excluded_rows=0,first_projected_sha256=None)
    def hook(module,args,output):
        value=mask_projected(output,mask);seen['calls']+=1
        if mask is not None and torch.count_nonzero(value.masked_select((~mask).expand_as(value))).item():
            raise ValueError('Excluded projected history content remains nonzero')
        if seen['first_projected_sha256'] is None:
            seen['first_projected_sha256']=tensor_sha(value);seen['excluded_rows']=0 if mask is None else int((~mask).sum())
        return value
    model=session['model'];handle=model.projection.register_forward_hook(hook)
    for processor in session['processors']:
        processor.text_policy='text_full';processor.prefix_rows=batch['prefix_rows']
        processor.text_rows=inputs['text_indices'].numel();processor.prefix_reads_current=not ids
    xv=noise['initial_video_noise'].to(device).clone();xa=noise['initial_audio_noise'].to(device).clone()
    if xv.shape!=inputs['hidden_states'].shape or xa.shape!=inputs['audio_hidden_states'].shape:raise ValueError('Target noise geometry changed')
    config=dict(policy=policy,case_id=case['case_id'],target_row=row,weights_spec=meta['weights_spec'],
        checkpoint_sha256=meta['weights_spec']['checkpoint_sha256'],prefix_sha256=meta['files_sha256']['PREFIX.pt'],
        input_manifest_path=str(root/'INPUT.json'),input_manifest_sha256=file_sha(root/'INPUT.json'),
        noise_manifest_path=str(root/'NOISE.pt'),noise_manifest_sha256=file_sha(root/'NOISE.pt'),
        video_noise_sha256=tensor_sha(xv),audio_noise_sha256=tensor_sha(xa),
        positions_fp32_sha256=hashes,projected_input_rows_sha256=tensor_sha(batch['history_rows']),NFE=50,CFG=1.,
        state_tokens_removed=False,mask_stage='after learned history projection',
        original_ids=meta['original_ids'],recent_ids=meta['recent_ids'],
        source_media_read=False,reference_target_media_available_to_worker=False,
        target_noise_source='original actual tensors; builder RNG is not used for sampling noise',
        generated_clock_only=True,formal_memory_quality_claim=False)
    (out/'RESOLVED_CONFIG.json').write_text(json.dumps(config,indent=2)+'\n')
    dist.barrier();tick=time.monotonic()
    try:
        with torch.inference_mode():
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
                    torch.cuda.synchronize();print(json.dumps(dict(event='native_return_probe',policy=policy,NFE=i+1,seconds=time.monotonic()-tick)),flush=True)
            torch.cuda.synchronize();dit=time.monotonic()-tick
            if seen['calls']!=50:raise ValueError('Projection intervention did not cover every NFE')
            v=unpack_video(xv,7,48,84)[0].to(torch.bfloat16).cpu()
            audio=xa[0].reshape(2,row['audio_latent_end']-row['audio_latent_start'],32).permute(0,2,1).to(torch.bfloat16).cpu()
            if not torch.isfinite(v).all() or not torch.isfinite(audio).all():raise ValueError('Nonfinite return output')
            archive=out/'TARGET_GENERATED.pt';torch.save(dict(video=v,audio=audio,positions_fp32=positions),archive)
    finally:handle.remove()
    # Only this committed target chunk is decoded, with the same end padding
    # for every condition. It is not a complete episode or latency measurement.
    with torch.inference_mode():
        vae=session['vae'];vae.to(device);decoder=FixedChunkVideoDecoder(vae,device);video=out/'target_chunk.mp4'
        with FrameWriter(video,24.) as writer:
            first=decoder.push(v[None].to(device));writer.append(first.cpu())
            tail=decoder.flush()
            if tail is not None:writer.append(tail.cpu())
            writer.close();artifact=writer.artifact()
        if artifact['frames']!=22:raise ValueError('First native target chunk must decode22frames')
        vae.cpu();torch.cuda.empty_cache();avae=session['audio_vae'];avae.to(device)
        mean=torch.tensor(avae.config.latents_mean,device=device).view(1,-1,1);std=torch.tensor(avae.config.latents_std,device=device).view(1,-1,1)
        pcm=avae.decode(audio.to(device)*std+mean,return_dict=False)[0].float().permute(1,0,2)[0].clamp(-1,1).cpu()
        desired=round(22/24.*32000);actual=pcm.shape[-1]
        if actual<desired:pcm=torch.nn.functional.pad(pcm,(0,desired-actual))
        wav=out/'target_chunk.wav'
        with wave.open(str(wav),'wb') as stream:
            stream.setnchannels(2);stream.setsampwidth(2);stream.setframerate(32000)
            stream.writeframes((pcm[:,:desired].T*32767).round().short().contiguous().numpy().tobytes())
        mux=out/'target_chunk.mux.mp4'
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-nostdin','-y','-i',str(video),'-i',str(wav),
            '-c:v','copy','-c:a','aac','-t',str(22/24.),str(mux)],check=True,timeout=120)
        mux.replace(video);artifact.update(sha256=file_sha(video),audio_muxed=True)
        avae.cpu();torch.cuda.empty_cache()
    value=dict(config=config,resolved_config_path=str(out/'RESOLVED_CONFIG.json'),projection_intervention=seen,artifact=artifact,generated_archive=str(archive),
        generated_archive_sha256=file_sha(archive),target_video_sha256=tensor_sha(v),target_audio_sha256=tensor_sha(audio),
        denoise_seconds=dit,generation_and_export_seconds=time.monotonic()-generation_started,
        model_load_seconds=session['load_seconds'],model_load_scope='shared two-rank session; do not sum across outputs',
        complete_episode=False,single_return_chunk_only=True,
        decode_policy='isolated22frame_chunk_with_common_end_padding_not_original_complete_episode_decode',
        audio_decode=dict(actual_samples=actual,delivered_samples=desired,trimmed_samples=max(0,actual-desired),padded_samples=max(0,desired-actual)),
        audio_encoder_lookahead_sec=0.,audio_future_context_allowed=False,audio_valid_mask_after_encoder='N/A: generated audio, no source encoding',
        hardware_host=socket.gethostname(),hardware_gpu=torch.cuda.get_device_name(device),formal_memory_quality_claim=False)
    temporary=out/'RESULT.tmp';temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(out/'RESULT.json')
    dist.barrier()


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);a=p.parse_args();manifest=json.loads(Path(a.manifest).read_text())
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=2:raise ValueError('Native probe uses two FSDP ranks per session')
    torch.cuda.set_device(local);torch.set_num_threads(2);dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=60))
    try:
        for tasks in manifest['rounds']:
            if len(tasks)!=2:raise ValueError('Equal forward counts required')
            for task in tasks:
                meta=json.loads((Path(task['input'])/'INPUT.json').read_text())
                if meta['weights_spec']!=manifest['weights_spec']:raise ValueError('Probe/model checkpoint or weight paths differ')
        session=load_session(manifest['weights_spec'],torch.device('cuda',local))
        for tasks in manifest['rounds']:
            if len(tasks)!=2:raise ValueError('Equal forward counts required')
            run(session,tasks[rank],torch.device('cuda',local))
    except BaseException:
        import traceback
        try:traceback.print_exc()
        except OSError:pass
        os._exit(1)
    else:dist.destroy_process_group()


if __name__=='__main__':main()
