"""H3 teacher/student rung substitution with write-once per-model video KV.

All arms use the published Stage1B sampler and attention contract. Dense H3
weights are explicitly evaluated under that causal contract; this is not a
native Dense oracle. The four-step grid never reaches video sigma < 0.8.
"""
import argparse
import datetime
import gc
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
# Bind before importing Diffusers/Triton and the shared loaders: otherwise
# module import side effects can create every rank's context on logical GPU0.
if 'LOCAL_RANK' in os.environ:
    import subprocess
    local_rank=int(os.environ['LOCAL_RANK'])
    if local_rank==0:
        preflight=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,memory.free',
            '--format=csv'],capture_output=True,text=True,timeout=20)
        print('GPU_BEFORE_CUDA_INIT\n'+preflight.stdout+preflight.stderr,flush=True)
        Path(os.environ['INFER_OUTPUT_DIR'],'GPU_BEFORE_CUDA_INIT.txt').write_text(preflight.stdout+preflight.stderr)
    torch.cuda.set_device(local_rank)
from train_h3_v2_local_history_ddp import bootstrap
bootstrap()
from h3_shared_loader import load_session
from h3_weights_interface_diagnostic import (
    file_sha, tensor_sha, install_stage1b_processors, stage1b_chunk,
    stage1b_clock, stage1b_clocks, stage1b_context, stage1b_kv_samples,
    stage1b_check_samples, stage1b_decode_long)
from h3_stage1_clock_probe import VideoKVBank


ARMS = {
    'student_all': ('student',) * 4,
    'teacher_high': ('teacher', 'teacher', 'student', 'student'),
    'teacher_lower': ('student', 'student', 'teacher', 'teacher'),
    'teacher_all': ('teacher',) * 4,
}
CHUNKS = ((0, 7), (7, 12), (12, 17), (17, 22))


def grid():
    result = []
    for i in range(4):
        tv, ta = stage1b_clocks(1 - i / 4)
        result.append(dict(rung=i, video_clean_time=tv, audio_clean_time=ta,
                           video_sigma=1-tv, audio_sigma=1-ta))
    return result


def atomic(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def same_across_ranks(value):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    if any(v != values[0] for v in values):
        raise ValueError('Replicated ranks have different requests/noise/positions')


def load_models(request, device, rank):
    sessions = {}
    for role in ('student', 'teacher'):
        spec = request[role + '_weights_spec']
        sessions[role] = load_session(spec, device, dense=True,
            load_decoders=rank == 0 and role == 'student',
            stage1b_current_only=True, dense_oracle=role == 'teacher', force_reshard=True)
        blocks=[]
        for wrapped in sessions[role]['model'].transformer.transformer_blocks:
            block=wrapped
            while hasattr(block,'_checkpoint_wrapped_module'):block=block._checkpoint_wrapped_module
            blocks.append(block)
        sessions[role]['blocks']=blocks
        sessions[role]['original_processors']=[b.attn.processor for b in blocks]
        install_stage1b_processors(sessions[role]['model'])
    return sessions


def prompt_for_model(batch, model):
    from torch.distributed.tensor import DTensor, Replicate, distribute_tensor
    x = batch['inputs']
    weight = model.transformer.context_embedder.weight
    if (isinstance(weight, DTensor) and isinstance(x['encoder_hidden_states'], DTensor)
        and x['encoder_hidden_states'].device_mesh != weight.device_mesh):
        x['encoder_hidden_states'] = x['encoder_hidden_states'].full_tensor()
    if isinstance(weight, DTensor) and not isinstance(x['encoder_hidden_states'], DTensor):
        x['encoder_hidden_states'] = distribute_tensor(x['encoder_hidden_states'],
                                                       weight.device_mesh, [Replicate()])


def run_arm(request, arm, sessions, prompt, device, rank, output):
    generator = torch.Generator(device=device).manual_seed(request['seed'])
    banks = {role: VideoKVBank(local_frames=9, sink_frames=3) for role in sessions}
    video_template = torch.zeros(24, 22, 48, 84, dtype=torch.bfloat16)
    audio_template = torch.zeros(2, 32, 122, dtype=torch.bfloat16)
    out = output / arm
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        atomic(out/'RESOLVED_CONFIG.json', dict(request=request, arm=arm,
            rung_roles=ARMS[arm], actual_grid=grid(), text_reads_history=False,
            video_history='separate fixed clean per-model KV; identical generated prefix',
            clean_writer_policy='each model writes current clean video once; no text/audio KV',
            old_history_queries_recomputed=0, force_reshard_after_forward=True,
            teacher_contract='Dense H3 weights under Stage1B causal rules; not native oracle',
            teacher_guidance=1., student_guidance=1., worker_sha256=file_sha(__file__)))
    records, archives = [], []
    began = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for ci, (start, end) in enumerate(CHUNKS):
            a_end = 122 if end == 22 else round((5 + 17*((end-2)//5))*40/24)
            batch = stage1b_chunk(video_template, audio_template, prompt['prompt_embeds'],
                prompt['text_token_tags'], generator, device, start, end, a_end)
            x = batch['inputs']
            positions_hash = tensor_sha(x['position_ids'])
            text_hash = tensor_sha(x['encoder_hidden_states'])
            prompt_for_model(batch, sessions['student']['model'])
            # Both sessions use the same one-dimensional default process group.
            video = torch.randn(x['hidden_states'].shape, device=device, generator=generator)
            audio = torch.randn(x['audio_hidden_states'].shape, device=device, generator=generator)
            initial_v, initial_a = video.cpu(), audio.cpu()
            hashes = [tensor_sha(video), tensor_sha(audio), positions_hash, text_hash]
            snapshots = {r: stage1b_kv_samples(b) for r,b in banks.items()}
            rungs = []
            for i, role in enumerate(ARMS[arm]):
                tv, ta = stage1b_clocks(1-i/4)
                x['hidden_states'], x['audio_hidden_states'] = video, audio
                stage1b_clock(batch, tv, ta)
                incoming_v, incoming_a = video.cpu(), audio.cpu()
                tick = time.monotonic()
                predictions={}
                for predictor, session in sessions.items():
                    prompt_for_model(batch,session['model'])
                    predictions[predictor]=session['model'].transformer(**x,
                        attention_mask=stage1b_context(batch, ci, start, end, banks[predictor]))
                pv, pa = predictions[role]
                torch.cuda.synchronize(device)
                clean_v = (video.float() + (1-tv)*pv.float()).to(torch.bfloat16)
                clean_a = (audio.float() + (1-ta)*pa.float()).to(torch.bfloat16)
                finite = bool(torch.isfinite(clean_v).all() and torch.isfinite(clean_a).all())
                if not finite:
                    raise ValueError(f'Nonfinite {arm} chunk={ci} rung={i}')
                rungs.append(dict(chunk=ci, role=role, **grid()[i],
                    input_video=incoming_v, input_audio=incoming_a,
                    clean_video=clean_v.cpu(), clean_audio=clean_a.cpu(),
                    video_velocity=pv.cpu(), audio_velocity=pa.cpu(), finite=finite,
                    same_input_video_velocity_relative_L2=float(
                        torch.linalg.vector_norm(predictions['student'][0].float()-predictions['teacher'][0].float()) /
                        torch.linalg.vector_norm(predictions['teacher'][0].float()).clamp_min(1e-12)),
                    same_input_audio_velocity_relative_L2=float(
                        torch.linalg.vector_norm(predictions['student'][1].float()-predictions['teacher'][1].float()) /
                        torch.linalg.vector_norm(predictions['teacher'][1].float()).clamp_min(1e-12)),
                    forward_seconds=time.monotonic()-tick))
                if i < 3:
                    nv, na = stage1b_clocks(1-(i+1)/4)
                    ev = torch.randn(clean_v.shape, device=device, generator=generator)
                    ea = torch.randn(clean_a.shape, device=device, generator=generator)
                    hashes.extend([tensor_sha(ev), tensor_sha(ea)])
                    # Match the strict matched H9 long worker, including BF16 multiply.
                    video = (nv*clean_v + (1-nv)*ev).to(torch.bfloat16)
                    audio = (na*clean_a + (1-na)*ea).to(torch.bfloat16)
                else:
                    video, audio = clean_v, clean_a
                del predictions
            same_across_ranks(hashes)
            retained = {r: stage1b_check_samples(snapshots[r],b) for r,b in banks.items()}
            stage1b_clock(batch, 1., 1.)
            x['hidden_states'], x['audio_hidden_states'] = video.detach(), audio.detach()
            commit_seconds = {}
            for role, session in sessions.items():
                tick=time.monotonic()
                prompt_for_model(batch,session['model'])
                session['model'].transformer(**x,
                    attention_mask=stage1b_context(batch, ci, start, end, banks[role], banks[role]))
                torch.cuda.synchronize(device)
                commit_seconds[role]=time.monotonic()-tick
                stage1b_check_samples(snapshots[role], banks[role])
            if rank == 0:
                archive=out/f'chunk{ci:03d}_GENERATED.pt'
                torch.save(dict(video=video.cpu(), audio=audio.cpu(), chunk=ci,
                    initial_video=initial_v, initial_audio=initial_a, rungs=rungs), archive)
                archives.append(str(archive))
                record=dict(chunk=ci,start_latent=start,end_latent=end,finite=True,
                    noise_and_position_hashes=hashes, rung_roles=ARMS[arm],
                    KV_bytes_by_model={r:b.bytes for r,b in banks.items()},
                    retained_KV_samples_equal=True, retained_KV_sample_rows=retained,
                    KV_sample_scope='two dimensions of head0; not full checksum',
                    clean_writes_by_model={r:1 for r in banks},
                    old_history_queries_recomputed=0,clean_commit_seconds=commit_seconds,
                    video_sha256=tensor_sha(video),audio_sha256=tensor_sha(audio),
                    rungs=[{k:v for k,v in r.items() if not torch.is_tensor(v)} for r in rungs])
                records.append(record)
                with (out/'TIMING.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps(dict(event='noise_arm_chunk_complete',arm=arm,chunk=ci)),flush=True)
            del snapshots,batch,x,video,audio,pv,pa,clean_v,clean_a,rungs
        del banks
        gc.collect(); torch.cuda.empty_cache(); dist.barrier()
        media_error=[None]
        if rank == 0:
            try:
                media=stage1b_decode_long(sessions['student'],out,archives,73,device)
                result=dict(status='complete_visual_review_pending',arm=arm,
                    checkpoint_step=sessions['student']['checkpoint_step'],
                    checkpoint_sha256=request['student_weights_spec']['checkpoint_sha256'],
                    NFE=4,chunks=4,frames=73,FPS=24,records=records,**media,
                    wall_seconds=time.monotonic()-began,
                    peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                    old_history_queries_recomputed=0,quality_verified=False,
                    teacher_native_dense_oracle=False,
                    lower_half_video_sigma_range=[0.8,12*.5/(1+11*.5)])
                atomic(out/'RESULT.json',result)
            except Exception as exc:
                media_error[0]=f'{type(exc).__name__}: {exc}'
                atomic(out/'EXPORT_ERROR.json',dict(error=media_error[0],latents_preserved=True))
        dist.broadcast_object_list(media_error,src=0)
        if media_error[0] is not None:raise RuntimeError(media_error[0])
    dist.barrier()
    return records


def decode_trace_controls(request, session, device, rank, output):
    if rank != 0:return
    # Actual training score predictions, not fresh student videos named fake.
    for case in request.get('training_trace_controls', []):
        path=Path(case['path'])
        if file_sha(path)!=case['sha256']:raise ValueError('Training trace changed')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        for name in ('student_generated','teacher_score_clean','fake_score_clean'):
            if name=='student_generated':v,a=saved['video'],saved['audio']
            else:
                prefix='real' if name=='teacher_score_clean' else 'fake'
                v=saved['score_diagnostics'][prefix+'_clean_video']
                a=saved['score_diagnostics'][prefix+'_clean_audio']
            out=output/'training_traces'/case['name']/name
            out.mkdir(parents=True,exist_ok=True)
            # Decoder handles one full 22-latent span; retain original float32
            # source tensors in the training archive, decode in native BF16.
            archive=out/'TRACE_DECODE_INPUT.pt'
            torch.save(dict(video=v.to(torch.bfloat16),audio=a.to(torch.bfloat16)),archive)
            with torch.inference_mode():media=stage1b_decode_long(session,out,[str(archive)],73,device)
            atomic(out/'RESULT.json',dict(status='training_trace_decoded_pending_visual_review',
                role=name,source_path=str(path),source_sha256=case['sha256'],
                source_prompt_id=saved['prompt_id'],score=saved['score'],
                source_exit_index=saved['exit_index'],**media,
                score_prediction_is_not_sampler_rollout=name!='student_generated',
                critic_loss_exact_fresh_rollout_saved=False,
                float32_to_bfloat16_for_vae_decode=True))


def native_teacher_control(request, sessions, prompt, device, rank, output):
    """Separate 50-NFE native Dense health control; not a matched-arm score."""
    if request.get('native_teacher_control_NFE') != 50:return
    from src.inference.render import generate_latents
    teacher=sessions['teacher']
    for block,processor in zip(teacher['blocks'],teacher['original_processors']):
        block.attn.set_processor(processor)
        block._segmented_modulation_enabled=False
    class NativeProxy:
        def __getattr__(self,name):return getattr(teacher['model'].transformer,name)
        def __call__(self,**inputs):
            batch={'inputs':dict(inputs)}
            prompt_for_model(batch,teacher['model'])
            return teacher['model'].transformer(**batch['inputs'])
    began=time.monotonic();timings=[]
    with torch.inference_mode():
        v,a=generate_latents(NativeProxy(),prompt['prompt_embeds'].to(device),
            prompt['text_token_tags'],73,50,request['seed'],device,step_seconds=timings)
        if not bool(torch.isfinite(v).all() and torch.isfinite(a).all()):
            raise ValueError('Native Dense teacher control is nonfinite')
        error=[None]
        if rank==0:
            try:
                out=output/'native_dense_teacher_50nfe';out.mkdir(exist_ok=True)
                packed=v.reshape(1,24,22,24,2,42,2).permute(0,2,3,5,1,4,6).reshape(1,22176,96)
                packed_audio=a.permute(0,2,1).reshape(1,244,32)
                archive=out/'NATIVE_LATENTS.pt'
                torch.save(dict(video=packed.to(torch.bfloat16).cpu(),audio=packed_audio.to(torch.bfloat16).cpu()),archive)
                media=stage1b_decode_long(sessions['student'],out,[str(archive)],73,device)
                atomic(out/'RESULT.json',dict(status='native_dense_teacher_health_pending_visual_review',
                    NFE=50,frames=73,FPS=24,finite=True,CFG=1.,**media,
                    matched_four_rung_intervention=False,full_document_bidirectional_attention=True,
                    causal_streaming_result=False,past_history_recomputed=False,
                    initial_noise_does_not_match_chunk_sampler=True,
                    per_step_seconds=timings,wall_seconds=time.monotonic()-began))
            except Exception as exc:error[0]=f'{type(exc).__name__}: {exc}'
        dist.broadcast_object_list(error,src=0)
        if error[0]:raise RuntimeError(error[0])
    dist.barrier()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--request',required=True);args=ap.parse_args()
    path=Path(args.request);request=json.loads(path.read_text())
    if (request.get('mode')!='h3_teacher_rung_diagnostic' or request.get('NFE')!=4
        or request.get('chunks')!=4 or request.get('history_recomputation_allowed') is not False
        or request.get('attention_contract')!='STAGE1B_CURRENT_ONLY_V1'):
        raise ValueError('Explicit unchanged Stage1B four-rung diagnostic required')
    arms=request['arms']
    if not arms or any(a not in ARMS for a in arms) or len(arms)!=len(set(arms)):
        raise ValueError('Invalid arm list')
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);torch.set_num_threads(2);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=30),device_id=device)
    try:
        if dist.get_world_size()!=8:raise ValueError('One eight-rank FSDP group required')
        same_across_ranks(file_sha(path))
        error=[None]
        if rank==0:
            try:
                spec=request['student_weights_spec']
                if file_sha(spec['checkpoint'])!=spec['checkpoint_sha256']:
                    raise ValueError('Student checkpoint hash differs from handoff')
            except Exception as exc:error[0]=repr(exc)
        dist.broadcast_object_list(error,src=0)
        if error[0]:raise ValueError(error[0])
        prompt=torch.load(request['prompt_path'],map_location='cpu',weights_only=True)
        if (prompt['prompt']!=request['prompt'] or file_sha(request['prompt_path'])!=request['prompt_file_sha256']
            or tensor_sha(prompt['prompt_embeds'])!=request['prompt_tensor_sha256']):
            raise ValueError('Frozen prompt mismatch')
        sessions=load_models(request,device,rank)
        reference=None;out=Path(os.environ['INFER_OUTPUT_DIR'])
        for arm in arms:
            records=run_arm(request,arm,sessions,prompt,device,rank,out)
            if rank==0:
                hashes=[r['noise_and_position_hashes'] for r in records]
                if reference is None:reference=hashes
                elif hashes!=reference:raise ValueError('Input noise/positions changed between arms')
        decode_trace_controls(request,sessions['student'],device,rank,out)
        dist.barrier()
        native_teacher_control(request,sessions,prompt,device,rank,out)
        if rank==0:atomic(out/'RUN_SUMMARY.json',dict(status='complete_visual_review_pending',
            arms=arms,request_sha256=file_sha(path),actual_grid=grid(),worker_sha256=file_sha(__file__),
            teacher_native_dense_oracle=False,source_reference_commit=request['source_reference_commit'],
            student_load_seconds=sessions['student']['load_seconds'],
            teacher_load_seconds=sessions['teacher']['load_seconds']))
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
