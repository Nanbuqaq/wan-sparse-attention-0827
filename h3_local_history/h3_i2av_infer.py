"""Native first-frame H3 reference, one loaded FSDP session for many requests."""
import argparse,datetime,gc,hashlib,inspect,json,os,socket,subprocess,time,wave
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def tensor_sha(x):
    import torch
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def validate_manifest(manifest):
    import torch
    from h3_i2av_contract import validate,validate_cache
    if manifest.get('policy')!='native_single_clip_I2AV' or manifest.get('NFE')!=50 or manifest.get('CFG')!=1.:
        raise ValueError('Explicit native I2AV 50-NFE/CFG1 protocol required')
    expected=[]
    for group in manifest['rounds']:
        mode=group['execution_mode'];tasks=group['tasks']
        if (mode=='independent_requests' and len(tasks)!=2) or (mode=='cooperative_FSDP_one_request' and len(tasks)!=1):
            raise ValueError('Declare two independent requests or one cooperative request, without padding the denominator')
        if mode not in ['independent_requests','cooperative_FSDP_one_request']:raise ValueError('Unknown execution mode')
        for task in tasks:
            validate(task)
            if task['conditioning']['model_contract_sha256']!=manifest['model_contract_sha256']:raise ValueError('Model contract differs')
            if sha(task['cache_path'])!=task['cache_sha256']:raise ValueError('Encoded image/text cache changed')
            validate_cache(task,torch.load(task['cache_path'],map_location='cpu',weights_only=True))
            image=task['conditioning']['image']
            if sha(image['path'])!=image['sha256']:raise ValueError('Benchmark image changed')
            expected.append(task['task_id'])
    if len(set(expected))!=len(expected):raise ValueError('A logical I2AV request was duplicated')
    return expected


def render(session,task,spec,mode):
    import torch
    import torch.distributed as dist
    from src.inference.render import generate_latents
    from h3_fixed_stream_decode import FixedChunkVideoDecoder
    from h3_frame_writer import FrameWriter
    import imageio_ffmpeg
    rank=dist.get_rank();device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    cooperative=mode=='cooperative_FSDP_one_request';publisher=not cooperative or rank==0
    out=Path(task['output']);out.mkdir(parents=True,exist_ok=True)
    result_file=out/'RESULT.json'
    if result_file.exists():raise FileExistsError('Completed native I2AV request must be reused, not overwritten')
    cache=torch.load(task['cache_path'],map_location='cpu',weights_only=True)
    conditions=(['first'],[cache['condition_latents'][0].to(device)])
    canvas=task['conditioning']['canvas'];initial={};observed={}
    def save_noise(video,audio):
        initial.update(initial_video_noise=video.detach().cpu().clone(),initial_audio_noise=audio.detach().cpu().clone())
    def save_condition(value):observed.update(value)
    resolved=out/f'rank{rank}_RESOLVED_CONFIG.json'
    atomic_json(resolved,dict(task=task,weights_spec=spec,NFE=50,CFG=1.,execution_mode=mode,
                             native_attention='full_current_document_no_generated_history',FSDP_world_size=dist.get_world_size(),
                             audio_encoder_status='not_invoked_no_input_audio',audio_encoder_lookahead_sec=0.,audio_future_context_allowed=False,
                             model_contract_sha256=task['conditioning']['model_contract_sha256']))
    torch.cuda.reset_peak_memory_stats(device);dist.barrier();started=time.monotonic();steps=[]
    with torch.inference_mode():
        video,audio=generate_latents(session['model'],cache['prompt_embeds'].to(device),cache['text_token_tags'],
            task['num_frames'],50,task['seed'],device,step_seconds=steps,conditions=conditions,
            height=canvas['height'],width=canvas['width'],initial_noise_callback=save_noise,conditioning_callback=save_condition)
    if len(steps)!=50 or video.shape!=(1,24,task['video_latent_frames'],canvas['height']//16,canvas['width']//16):
        raise ValueError('Native I2AV geometry/NFE differs from requested contract')
    if not torch.isfinite(video).all() or not torch.isfinite(audio).all():raise ValueError('Nonfinite native I2AV output')
    peak_dit=torch.cuda.max_memory_allocated(device)/2**30
    noise_file=out/f'rank{rank}_INITIAL_NOISE_AND_LAYOUT.pt'
    torch.save(dict(**initial,**observed,task_id=task['task_id'],conditioning_sha256=task['conditioning_sha256']),noise_file)
    v=video.detach().cpu();a=audio.detach().cpu();del video,audio,conditions;gc.collect();torch.cuda.empty_cache()
    noise_manifest=dict(video_noise_sha256=tensor_sha(initial['initial_video_noise']),audio_noise_sha256=tensor_sha(initial['initial_audio_noise']),
                        condition_noise_sha256=[tensor_sha(x) for x in observed['condition_noises']],
                        pinned_condition_rows_sha256=[tensor_sha(x) for x in observed['pinned_condition_rows']],
                        positions_sha256=tensor_sha(observed['position_ids']),num_condition_rows=observed['num_condition_rows'],
                        draw_order='native_condition_noise_then_target_video_noise_then_target_audio_noise')
    if cooperative:
        peers=[None]*dist.get_world_size();dist.all_gather_object(peers,noise_manifest)
        if any(peer!=noise_manifest for peer in peers):raise ValueError('Cooperative ranks used different actual inputs')
    else:peers=None
    if publisher:
        archive=out/'GENERATED_LATENTS.pt';torch.save(dict(video=v,audio=a,task_id=task['task_id']),archive)
        vae=session['vae'];vae.to(device);decode_start=time.monotonic()
        with torch.inference_mode():decoded=FixedChunkVideoDecoder(vae,device).decode(v.to(device))
        path=out/'video.mp4';temporary=out/'video.partial.mp4'
        with FrameWriter(temporary,24.) as writer:
            writer.append(decoded);writer.close();artifact=writer.artifact()
        del decoded;vae.cpu();gc.collect();torch.cuda.empty_cache()
        if artifact['frames']!=task['num_frames']:raise ValueError('Native video frame count changed')
        avae=session['audio_vae'];avae.to(device)
        mean=torch.tensor(avae.config.latents_mean,device=device).view(1,-1,1);std=torch.tensor(avae.config.latents_std,device=device).view(1,-1,1)
        with torch.inference_mode():pcm=avae.decode(a.to(device)*std+mean,return_dict=False)[0].float().permute(1,0,2)[0].clamp(-1,1).cpu()
        avae.cpu();torch.cuda.empty_cache();decoded_samples=pcm.shape[-1];desired=round(task['num_frames']/24*32000)
        if pcm.shape[0]!=2:raise ValueError('Native soundtrack must be stereo')
        def write_wave(target,value):
            with wave.open(str(target),'wb') as stream:
                stream.setnchannels(2);stream.setsampwidth(2);stream.setframerate(32000)
                stream.writeframes((value.T*32767).round().short().contiguous().numpy().tobytes())
        write_wave(out/'native_audio.wav',pcm)
        delivered=torch.nn.functional.pad(pcm,(0,max(0,desired-decoded_samples)))[:,:desired]
        write_wave(out/'audio.wav',delivered)
        mux=out/'mux.partial.mp4'
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-v','error','-y','-i',str(temporary),'-i',str(out/'audio.wav'),
                        '-c:v','copy','-c:a','aac','-t',str(task['num_frames']/24),str(mux)],check=True,timeout=120)
        mux.replace(path);temporary.rename(out/'video_only.mp4');artifact.update(path=str(path),sha256=sha(path),audio_muxed=True)
        result=dict(status='native_I2AV_generated_v1',task_id=task['task_id'],case_id=task['case_id'],seed=task['seed'],artifact=artifact,
                    resolved_config_path=str(resolved),resolved_config_sha256=sha(resolved),conditioning_sha256=task['conditioning_sha256'],
                    conditioning_cache_sha256=task['cache_sha256'],noise_manifest=noise_manifest,noise_file=str(noise_file),noise_file_sha256=sha(noise_file),
                    model_contract_sha256=task['conditioning']['model_contract_sha256'],checkpoint_sha256=spec.get('checkpoint_sha256'),
                    weights_kind=spec['kind'],checkpoint_step=session['checkpoint_step'],model_load_seconds=session['load_seconds'],
                    model_load_scope='shared session, do not sum per-video load_seconds',
                    session_load_proof=session['load_proof'],
                    generated_latents=str(archive),generated_latents_sha256=sha(archive),
                    audio_wav=str(out/'audio.wav'),audio_wav_sha256=sha(out/'audio.wav'),native_audio_wav=str(out/'native_audio.wav'),
                    native_audio_wav_sha256=sha(out/'native_audio.wav'),
                    native_audio_samples=decoded_samples,delivered_audio_samples=desired,audio_trimmed_samples=max(0,decoded_samples-desired),
                    audio_padded_samples=max(0,desired-decoded_samples),audio_valid_intervals_samples=[[0,min(decoded_samples,desired)]],
                    NFE=50,CFG=1.,dit_seconds=sum(steps),per_NFE_seconds=steps,decode_mux_seconds=time.monotonic()-decode_start,
                    generation_export_seconds=time.monotonic()-started,peak_DiT_allocated_gib=peak_dit,
                    physical_GPU_count=dist.get_world_size(),execution_mode=mode,logical_output_count=1,cooperative_input_checks=peers,
                    physical_GPU_count_scope='one_shared_FSDP_session; two logical requests share it in independent_requests mode',
                    worker_source_sha256=sha(__file__),native_render_source_sha256=sha(inspect.getfile(inspect.unwrap(generate_latents))),
                    native_render_wrapper_source_sha256=sha(inspect.getfile(generate_latents)),
                    native_render_source_binding='unwrapped_callable_file_v2',
                    hardware_host=socket.gethostname(),GPU=torch.cuda.get_device_name(device),torch_version=torch.__version__,
                    comparison_track='nonstreaming_h3_native_I2AV_reference' if spec['kind']=='base' else 'nonstreaming_h3_I2AV_transfer_diagnostic',native_I2VA_used=True,
                    generated_clock=True,source_time_admitted=False,streaming_quality_claim=False,formal_memory_quality_claim=False,
                    audio_encoder_lookahead_sec=0.,audio_future_context_allowed=False,audio_valid_mask_after_encoder='N/A: no input audio encoder')
        atomic_json(result_file,result)
    del v,a,initial,observed;gc.collect();torch.cuda.empty_cache();dist.barrier()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--validate-inputs-only',action='store_true');a=parser.parse_args()
    from train_h3_v2_local_history_ddp import bootstrap
    bootstrap()
    manifest=json.loads(Path(a.manifest).read_text());expected=validate_manifest(manifest)
    if a.validate_inputs_only:print(json.dumps(dict(logical_requests=len(expected),neural_forward=False)));return
    if not os.environ.get('INFER_CODE_DIR'):raise RuntimeError('Native I2AV inference must use official InferHub')
    import torch
    import torch.distributed as dist
    from h3_shared_loader import load_session
    from h3_i2av_encode import verify_assets
    verify_assets(Path(os.environ['INFER_WEIGHTS_DIR']),manifest['model_contract_sha256'])
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);torch.set_num_threads(2)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=60))
    if dist.get_world_size()!=2:raise ValueError('This native I2AV sampler uses2-rank parameter sharding')
    try:
        session=load_session(manifest['weights_spec'],torch.device('cuda',local),dense=True)
        load=Path(manifest['session_output'])/f'rank{rank}_LOAD.json'
        atomic_json(load,dict({k:v for k,v in session.items() if k not in ['model','processors','vae','audio_vae','decoder']},
                              model_contract_sha256=manifest['model_contract_sha256']))
        session['load_proof']=dict(path=str(load),sha256=sha(load))
        for group in manifest['rounds']:
            task=group['tasks'][0 if group['execution_mode']=='cooperative_FSDP_one_request' else rank]
            render(session,task,manifest['weights_spec'],group['execution_mode'])
    finally:dist.destroy_process_group()

if __name__=='__main__':main()
