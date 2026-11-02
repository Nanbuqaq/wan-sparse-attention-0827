"""Keep the trained four-step student prefix; finish on a fine teacher grid.

Teacher grid points come from the native 50-evaluation scheduler. A handoff
point is inserted when absent so both video/audio clocks match the actual
noisy student state. Teacher uses the declared fixed-KV causal interface.
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
if 'LOCAL_RANK' in os.environ:
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
from train_h3_v2_local_history_ddp import bootstrap
bootstrap()
from h3_teacher_rung_diagnostic import (
    CHUNKS, atomic, file_sha, tensor_sha, same_across_ranks, load_models,
    prompt_for_model, stage1b_chunk, stage1b_context, stage1b_clock,
    stage1b_clocks, stage1b_kv_samples, stage1b_check_samples,
    stage1b_decode_long, VideoKVBank)


def clocks_from_u(u):
    tv,ta=stage1b_clocks(u)
    return dict(u=float(u),video_time=tv,audio_time=ta,video_sigma=1-tv,audio_sigma=1-ta)


def teacher_nodes(student_steps,teacher_nfe=50):
    if student_steps not in (0,1,2,3) or teacher_nfe!=50:
        raise ValueError('Use 0/1/2/3 student steps and the 50-evaluation teacher grid')
    # Use float32 points just like MiniMaxH3Scheduler.set_timesteps(51).
    raw=torch.linspace(1.,0.,teacher_nfe+1,dtype=torch.float32).tolist()
    start=1-student_steps/4
    return [start]+[float(u) for u in raw if u<start-1e-7]


def clean_prediction(video,audio,prediction,tv,ta):
    pv,pa=prediction
    return ((video.float()+(1-tv)*pv.float()).to(torch.bfloat16),
            (audio.float()+(1-ta)*pa.float()).to(torch.bfloat16))


def declared_schedule(k):
    if k==4:
        return dict(student_steps=4,teacher_steps=0,handoff=None,teacher_nodes=[])
    nodes=teacher_nodes(k)
    return dict(student_steps=k,teacher_steps=len(nodes)-1,
        handoff=clocks_from_u(nodes[0]),teacher_nodes=[clocks_from_u(u) for u in nodes])


def native_clock_value(u,shift):
    # Match the native scheduler's float32 shifted linspace arithmetic.
    raw=torch.tensor(u,dtype=torch.float32)
    return float(1-shift*raw/(1+(shift-1)*raw))


def teacher_clock_pairs(k):
    nodes=teacher_nodes(k);pairs=[]
    for j,u in enumerate(nodes):
        if j==0 and k:
            # The student uses Python-float warp values rounded by set_clock.
            # Reuse those exact values at the handoff; no approximate lookup.
            tv,ta=stage1b_clocks(u)
        else:tv,ta=native_clock_value(u,12),native_clock_value(u,3)
        pairs.append((tv,ta))
    return pairs


def freeze_chunk_noise(batch,generator,device):
    x=batch['inputs']
    v=torch.randn(x['hidden_states'].shape,device=device,generator=generator)
    a=torch.randn(x['audio_hidden_states'].shape,device=device,generator=generator)
    # Draw the entire four-step tape for every arm, including unused suffixes.
    # Subsequent chunks therefore use the same initial noise across all k.
    tape=[]
    for _ in range(3):
        tape.append((torch.randn(v.shape,device=device,generator=generator),
                     torch.randn(a.shape,device=device,generator=generator)))
    return v,a,tape


def rollout(request,k,sessions,prompt,device,rank,out_root):
    name='student_4_full' if k==4 else 'teacher_50_full' if k==0 else f'student_{k}_then_teacher'
    out=out_root/name
    used_roles=('student',) if k==4 else ('teacher',) if k==0 else ('student','teacher')
    banks={r:VideoKVBank(9,3) for r in used_roles}
    g=torch.Generator(device=device).manual_seed(request['seed'])
    vt=torch.zeros(24,22,48,84,dtype=torch.bfloat16)
    at=torch.zeros(2,32,122,dtype=torch.bfloat16)
    records=[];archives=[];began=time.monotonic();torch.cuda.reset_peak_memory_stats(device)
    schedule=declared_schedule(k)
    if rank==0:
        out.mkdir(parents=True,exist_ok=True)
        atomic(out/'RESOLVED_CONFIG.json',dict(request=request,student_prefix_steps=k,
            schedule=schedule,teacher_fine_grid='native50 raw grid with exact handoff inserted',
            teacher_interface='Dense H3 weights under fixed-KV current-only Stage1B causal rules',
            text_reads_history=False,text_KV_committed=False,audio_KV_committed=False,
            historical_latents_recomputed=False,source_sha256=file_sha(__file__)))
    with torch.inference_mode():
        for ci,(start,end) in enumerate(CHUNKS):
            ae=122 if end==22 else round((5+17*((end-2)//5))*40/24)
            batch=stage1b_chunk(vt,at,prompt['prompt_embeds'],prompt['text_token_tags'],g,device,start,end,ae)
            x=batch['inputs'];input_hashes=dict(positions=tensor_sha(x['position_ids']),text=tensor_sha(x['encoder_hidden_states']))
            video,audio,tape=freeze_chunk_noise(batch,g,device)
            input_hashes.update(initial_video=tensor_sha(video),initial_audio=tensor_sha(audio),
                renoise=[dict(video=tensor_sha(v),audio=tensor_sha(a)) for v,a in tape])
            same_across_ranks(input_hashes)
            before={r:stage1b_kv_samples(b) for r,b in banks.items()}
            steps=[];handoff=None;handoff_tensors=None;student_inputs=[]
            for i in range(k):
                tv,ta=stage1b_clocks(1-i/4)
                x['hidden_states'],x['audio_hidden_states']=video,audio;stage1b_clock(batch,tv,ta)
                prompt_for_model(batch,sessions['student']['model'])
                student_inputs.append(dict(step=i,video=tensor_sha(video),audio=tensor_sha(audio)))
                tick=time.monotonic()
                pred=sessions['student']['model'].transformer(**x,
                    attention_mask=stage1b_context(batch,ci,start,end,banks['student']))
                cv,ca=clean_prediction(video,audio,pred,tv,ta)
                torch.cuda.synchronize(device)
                steps.append(dict(role='student',student_step=i,**clocks_from_u(1-i/4),
                    input_video_sha256=student_inputs[-1]['video'],input_audio_sha256=student_inputs[-1]['audio'],
                    clean_video_sha256=tensor_sha(cv),clean_audio_sha256=tensor_sha(ca),
                    seconds=time.monotonic()-tick))
                if i==3:video,audio=cv,ca
                else:
                    nv,na=stage1b_clocks(1-(i+1)/4);ev,ea=tape[i]
                    # The exact trained stochastic x0/re-noising formula.
                    video=(nv*cv+(1-nv)*ev).to(torch.bfloat16)
                    audio=(na*ca+(1-na)*ea).to(torch.bfloat16)
                del pred,cv,ca
            if k<4:
                nodes=teacher_nodes(k);times=teacher_clock_pairs(k)
                handoff=dict(student_steps=k,**clocks_from_u(nodes[0]),
                    video_sha256=tensor_sha(video),audio_sha256=tensor_sha(audio),
                    student_prefix_update_preserved=True,teacher_receives_noisy_latent=True)
                handoff_tensors=dict(video=video.cpu(),audio=audio.cpu())
                for j,(tv,ta) in enumerate(times[:-1]):
                    nv,na=times[j+1]
                    x['hidden_states'],x['audio_hidden_states']=video,audio;stage1b_clock(batch,tv,ta)
                    prompt_for_model(batch,sessions['teacher']['model'])
                    actual=x['timestep'][x['timestep_indices']]
                    if j==0:
                        assert tensor_sha(video)==handoff['video_sha256']
                        assert tensor_sha(audio)==handoff['audio_sha256']
                        assert actual[x['video_indices'][0]]==torch.tensor(tv,device=device,dtype=actual.dtype)
                        assert actual[x['audio_indices'][0]]==torch.tensor(ta,device=device,dtype=actual.dtype)
                    tick=time.monotonic()
                    pv,pa=sessions['teacher']['model'].transformer(**x,
                        attention_mask=stage1b_context(batch,ci,start,end,banks['teacher']))
                    video=video.float()+(nv-tv)*pv.float()
                    audio=audio.float()+(na-ta)*pa.float()
                    torch.cuda.synchronize(device)
                    steps.append(dict(role='teacher',teacher_step=j,u=nodes[j],video_time=tv,audio_time=ta,
                        next_video_time=nv,next_audio_time=na,video_sigma=1-tv,audio_sigma=1-ta,
                        seconds=time.monotonic()-tick))
                    del pv,pa
            video,audio=video.to(torch.bfloat16),audio.to(torch.bfloat16)
            if not bool(torch.isfinite(video).all() and torch.isfinite(audio).all()):
                raise RuntimeError('Nonfinite prefix/teacher result')
            for r,b in banks.items():stage1b_check_samples(before[r],b)
            x['hidden_states'],x['audio_hidden_states']=video.detach(),audio.detach();stage1b_clock(batch,1.,1.)
            for role in used_roles:
                prompt_for_model(batch,sessions[role]['model'])
                sessions[role]['model'].transformer(**x,
                    attention_mask=stage1b_context(batch,ci,start,end,banks[role],banks[role]))
                stage1b_check_samples(before[role],banks[role])
            torch.cuda.synchronize(device)
            rec=dict(chunk=ci,start=start,end=end,finite=True,noise_and_position_hashes=input_hashes,
                student_inputs=student_inputs,student_denoise_evaluations=k,
                teacher_denoise_evaluations=schedule['teacher_steps'],
                teacher_model_called=k<4,teacher_clean_commits=int(k<4),
                student_clean_commits=int(k>0),handoff=handoff,steps=steps,
                old_history_queries_recomputed=0,retained_KV_samples_equal=True,
                KV_bytes_by_role={r:b.bytes for r,b in banks.items()},
                video_sha256=tensor_sha(video),audio_sha256=tensor_sha(audio))
            if rank==0:
                archive=out/f'chunk{ci:03d}_GENERATED.pt'
                torch.save(dict(video=video.cpu(),audio=audio.cpu(),handoff=handoff_tensors,record=rec),archive)
                archives.append(str(archive));records.append(rec)
                with (out/'TIMING.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
                print(json.dumps(dict(event='prefix_completion_chunk_done',arm=name,chunk=ci,
                    student_calls=k,teacher_calls=schedule['teacher_steps'])),flush=True)
            del before,batch,x,video,audio,tape
        del banks
        gc.collect();torch.cuda.empty_cache();dist.barrier()
        error=[None]
        if rank==0:
            try:
                media=stage1b_decode_long(sessions['student'],out,archives,73,device)
                atomic(out/'RESULT.json',dict(status='complete_pending_visual_review',arm=name,
                    student_steps=k,teacher_steps=schedule['teacher_steps'],schedule=schedule,
                    records=records,frames=73,FPS=24,chunks=4,**media,
                    wall_seconds=time.monotonic()-began,checkpoint_sha256=request['student_weights_spec']['checkpoint_sha256'],
                    request_sha256=file_sha(request['request_path']),
                    peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                    teacher_native_full_document_attention=False,teacher_fine_base_NFE=50,
                    student_native_four_step_rule_preserved=True,
                    all_student_has_no_teacher_forward=k==4,
                    old_history_queries_recomputed=0,quality_verified=False))
            except Exception as exc:
                error[0]=repr(exc);atomic(out/'EXPORT_ERROR.json',dict(error=error[0],latents_saved=True))
        dist.broadcast_object_list(error,src=0)
        if error[0]:raise RuntimeError(error[0])
        dist.barrier()
    return records


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--request',required=True);a=ap.parse_args()
    path=Path(a.request);request=json.loads(path.read_text());request['request_path']=str(path)
    if not (request['mode']=='student_prefix_teacher_completion' and request['teacher_NFE']==50
        and request['student_NFE']==4 and request['history_recomputation_allowed'] is False):
        raise ValueError('Explicit 4-step student / 50-grid teacher completion required')
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);device=torch.device('cuda',local)
    torch.cuda.set_device(local);torch.set_num_threads(2)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=30),device_id=device)
    try:
        if dist.get_world_size()!=8:raise ValueError('Eight ranks required')
        same_across_ranks(file_sha(path))
        err=[None]
        if rank==0:
            spec=request['student_weights_spec']
            if file_sha(spec['checkpoint'])!=spec['checkpoint_sha256']:err[0]='Checkpoint hash mismatch'
        dist.broadcast_object_list(err,src=0)
        if err[0]:raise ValueError(err[0])
        prompt=torch.load(request['prompt_path'],map_location='cpu',weights_only=True)
        if (file_sha(request['prompt_path'])!=request['prompt_file_sha256']
            or tensor_sha(prompt['prompt_embeds'])!=request['prompt_tensor_sha256']
            or prompt['prompt']!=request['prompt']):raise ValueError('Prompt changed')
        sessions=load_models(request,device,rank)
        out=Path(os.environ['INFER_OUTPUT_DIR']);baseline=None;reference_noise=None;complete=[]
        for k in request['student_prefix_steps']:
            records=rollout(request,k,sessions,prompt,device,rank,out)
            if rank==0:
                hashes=[r['noise_and_position_hashes'] for r in records]
                if reference_noise is None:reference_noise=hashes
                elif hashes!=reference_noise:raise ValueError('Noise tape differs across prefix lengths')
                if k==4:baseline=records[0]['student_inputs']
                elif k>0 and baseline is not None:
                    if records[0]['student_inputs']!=baseline[:k]:
                        raise ValueError('First-chunk student prefix differs from four-step baseline')
                complete.append(dict(student_steps=k,teacher_steps=declared_schedule(k)['teacher_steps']))
        if rank==0:atomic(out/'RUN_SUMMARY.json',dict(status='complete_pending_visual_review',arms=complete,
            source_sha256=file_sha(__file__),request_sha256=file_sha(path),
            same_noise_tape_all_chunks=True,first_chunk_student_prefix_exact=True,
            teacher_load_seconds=sessions['teacher']['load_seconds'],
            student_load_seconds=sessions['student']['load_seconds']))
        dist.barrier()
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
