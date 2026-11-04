"""One complete prompt program through Agent 5's shared fixed-KV sampler."""
import argparse
import datetime
import hashlib
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
from h3_shared_stage1 import shared_modules,TransformerCallProxy
from h3_shared_kv_heads import require_same_request
from h3_buffered_video_decode import BufferedVideoDecoder
from h3_frame_writer import FrameWriter
from sample_demo import unpack_video
from h3_shared_request_contract import validate_shared_request


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class Observer:
    def __init__(self,strategy):
        self.block=None;self.calls=[];self.initial_noise=None
        self.strategy=strategy
    def begin(self,number):
        self.block=number;self.calls=[];self.initial_noise=None
    def before(self,inputs,context):
        if context.text_reads_history!=(self.strategy=='B'):
            raise RuntimeError('Executed text/history visibility differs from the requested A/B/C strategy')
        if any(s.block!=self.block for s in context.splits):
            raise RuntimeError('A committed history block was resent through the model')
        phase='commit' if context.write_bank is not None else 'denoise'
        if phase=='denoise' and self.initial_noise is None:
            self.initial_noise=dict(video_sha256=tensor_sha(inputs['hidden_states']),
                                    audio_sha256=tensor_sha(inputs['audio_hidden_states']))
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();self.calls.append((phase,start,end))
    def after(self,result):
        self.calls[-1][2].record()
    def timings(self):
        denoise=[a.elapsed_time(b)/1000 for phase,a,b in self.calls if phase=='denoise']
        commit=[a.elapsed_time(b)/1000 for phase,a,b in self.calls if phase=='commit']
        if len(denoise)!=50 or len(commit)!=1:
            raise RuntimeError('Expected fifty current-only calls plus one new-video clean commit')
        return dict(per_NFE_seconds=denoise,DiT_seconds=sum(denoise),clean_commit_seconds=commit[0])


def block_template(row,prompt,origin,device='cpu'):
    start=row['local_latent_start']
    frame0=row['shot_frame_start']
    count=int(row.get('latent_count', 7 if int(row['block']) == 0 else 5))
    local_frames=[17*(i//5)+(0,1,5,9,13)[i%5] for i in range(start,start+count)]
    positions=torch.empty((count,24,42,3))
    positions[:,:,:,0]=torch.tensor([origin+40*(frame0+f)/24 for f in local_frames])[:,None,None]
    positions[:,:,:,1]=torch.arange(24)[None,:,None]
    positions[:,:,:,2]=torch.arange(42)[None,None,:]
    a0,a1=row['audio_start'],row['audio_end']
    audio_pos=torch.zeros(2,a1-a0,3)
    audio_pos[:,:,0]=origin+torch.arange(a0,a1)[None]
    audio_pos[:,:,1]=torch.arange(2)[:,None]
    return dict(block=row['block'],video=torch.zeros(count*1008,96),audio=torch.zeros(2*(a1-a0),32),
                prompt=prompt['prompt_embeds'],video_pos=positions.reshape(-1,3),
                audio_pos=audio_pos.reshape(-1,3),audio_mask=torch.ones(2*(a1-a0),dtype=torch.bool),
                source_start=row['start_frame']/24,source_end=row['end_frame']/24)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--request',required=True)
    ap.add_argument('--preflight-only',action='store_true');ap.add_argument('--verify-checkpoints',action='store_true')
    args=ap.parse_args()
    requested=Path(args.request);request=json.loads(requested.read_text())
    dispatch=validate_shared_request(request)
    request_sha=hashlib.sha256(requested.read_bytes()).hexdigest()
    if args.preflight_only:
        checkpoint=Path(request['weights_spec']['checkpoint'])
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        if (payload.get('schema')!='h3_full_dit_inference_v1'
                or payload.get('attention_contract')!=dispatch['attention_contract']
                or payload.get('metadata',{}).get('args',{}).get('ablation_arm')!=dispatch['strategy']):
            raise ValueError('Checkpoint header strategy/contract differs from the sampler')
        del payload
        if args.verify_checkpoints:
            digest=hashlib.sha256()
            with checkpoint.open('rb') as stream:
                for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
            if digest.hexdigest()!=request['weights_spec']['checkpoint_sha256']:
                raise ValueError('Checkpoint content hash differs')
        for path,text in zip(request['prompt_paths'],request['prompts']):
            if torch.load(path,map_location='cpu',weights_only=True)['prompt']!=text:
                raise ValueError('Prompt text differs from its encoding')
        marker=requested.with_name('PREP_READY.json')
        value=dict(request_sha256=request_sha,dispatch=dispatch,
                   checkpoint_content_hash_verified=args.verify_checkpoints,
                   checkpoint_identity=[checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns])
        temporary=marker.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(marker)
        print('SHARED_ABC_PREFLIGHT_OK',flush=True)
        return
    if request.get('requires_verified_preflight'):
        proof=json.loads(requested.with_name('PREP_READY.json').read_text())
        checkpoint=Path(request['weights_spec']['checkpoint'])
        if (proof['request_sha256']!=request_sha or not proof['checkpoint_content_hash_verified']
                or proof['dispatch']!=dispatch
                or proof['checkpoint_identity']!=[checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns]):
            raise ValueError('Verified request/checkpoint changed after CPU prep')
    started=time.monotonic();torch.set_num_threads(2)
    rank,local=int(os.environ['RANK']),int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    # B-lane text reads committed history and can exceed 20 minutes on a
    # loaded host. The old watchdog timeout killed a valid collective and was
    # misreported as a model failure. InferHub's external timeout is separate.
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=40),device_id=device)
    out=Path(os.environ['INFER_OUTPUT_DIR'])
    try:
        if dist.get_world_size()!=8:raise ValueError('Use one eight-card group for this complete video')
        require_same_request(hashlib.sha256(requested.read_bytes()).hexdigest())
        code,source=shared_modules()
        session=load_session(request['weights_spec'],device)
        observer=Observer(request['strategy']);proxy=TransformerCallProxy(session['model'],observer)
        stream=code['h3_fixed_kv_runtime'].FixedKVStream(proxy,request['strategy'])
        generator=torch.Generator(device=device).manual_seed(request['seed'])
        prompts=[torch.load(p,map_location='cpu',weights_only=True) for p in request['prompt_paths']]
        for item,text in zip(prompts,request['prompts']):
            if item['prompt']!=text or not bool((item['text_token_tags']==1).all()):
                raise ValueError('Prompt encoding differs from the official input')
        origin=prompts[0]['prompt_embeds'].shape[0]
        video_path=out/'VIDEO.mp4';records=[];audio=[];snapshots=[];first_video_frames_seconds=None
        # Do not decode VAE inside the generation loop. Rank zero must stay in
        # lock-step with the other ranks; per-block VAE transfers made B's
        # text-history lane hit the NCCL watchdog while the other ranks waited.
        decoder=None
        writer=None
        if rank==0:
            (out/'RESOLVED_CONFIG.json').write_text(json.dumps(dict(request=request,
                source_manifest=source,weights_metadata=session['training_metadata'],
                GPU=torch.cuda.get_device_name(device),world_size=8,
                decoder='buffered_native_equivalent_v1',source_pts=None,
                data_clock='explicit_generated_24FPS_clock',formal_memory_quality_claim=False,
                offline_audio_decode=True,model_history_resets_at_codec_cut=False),indent=2)+'\n')
        with torch.inference_mode():
            for row in request['blocks']:
                tick=time.monotonic();observer.begin(row['block'])
                state=generator.get_state().cpu()
                require_same_request((row['block'],tensor_sha(state),request['seed']))
                block=block_template(row,prompts[row['prompt_index']],origin)
                video,pcm_latent,stats=stream.generate(block,generator,device,50)
                torch.cuda.synchronize();sample_wall=time.monotonic()-tick
                timing=observer.timings();delivery_start=time.monotonic()
                if rank==0:
                    # The shared runtime supports both fixed-5 ABC requests and
                    # the native startup7+regular5 geometry.  Derive the
                    # decoder row count from the frozen request; hard-coding 5
                    # silently turns a startup7 block into a reshape error,
                    # after which the other ranks wait at the barrier until
                    # NCCL times out.
                    latent_count = int(row.get('latent_count', 7 if int(row['block']) == 0 else 5))
                    latent=unpack_video(video,latent_count,48,84).to(torch.bfloat16)
                    sound=pcm_latent[0].reshape(2,-1,32).permute(0,2,1).to(torch.bfloat16)
                    snapshot=out/f'chunk{row["block"]:03d}_GENERATED.pt'
                    torch.save(dict(video=latent.cpu(),audio=sound.cpu(),generator_before=state,
                        generator_after=generator.get_state().cpu(),row=row,
                        initial_noise=observer.initial_noise),snapshot)
                    snapshots.append(str(snapshot));audio.append(sound.cpu())
                    record=dict(block=row['block'],start_sec=block['source_start'],end_sec=block['source_end'],
                        prompt_index=row['prompt_index'],prompt_sha256=hashlib.sha256(request['prompts'][row['prompt_index']].encode()).hexdigest(),
                        history_blocks=list(stats['read_past_blocks']),
                        history_video_tokens=len(stats['read_past_blocks'])*5040,
                        current_video_tokens=5040,cache_bytes_per_rank=stats['KV_bytes'],
                        sample_wall_seconds=sample_wall,**timing,
                        decode_and_snapshot_seconds=time.monotonic()-delivery_start,
                        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                        old_history_queries_recomputed=0,**observer.initial_noise)
                    record['effective_text_reads_history']=dispatch['effective_text_reads_history']
                    record['clean_commit_forwards']=stats['clean_commit_forwards']
                    records.append(record)
                    with (out/'TIMING.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                    print(json.dumps(record),flush=True)
                dist.barrier()
            if rank==0:
                if session['reshard']:session['vae'].to(device)
                decoder=BufferedVideoDecoder(session['vae'],device)
                writer=FrameWriter(video_path,24,encoder_threads=2)
                for snapshot in snapshots:
                    saved=torch.load(snapshot,map_location='cpu',weights_only=True)
                    decoded=decoder.push(saved['video'].to(device))
                    if decoded is not None:
                        writer.append(decoded)
                        if first_video_frames_seconds is None:first_video_frames_seconds=time.monotonic()-started
                tail=decoder.finish()
                if tail is not None:writer.append(tail)
                writer.close()
                if writer.frames!=request['actual_frames']:
                    raise ValueError('Decoded video length differs from the declared complete program')
                torch.save(dict(chunks=snapshots,actual_frames=writer.frames,source_manifest=source),out/'GENERATED_LATENT_MANIFEST.pt')
                audio_started=time.monotonic();al=torch.cat(audio,dim=2).to(device)
                if session['reshard']:
                    session['vae'].cpu();session['audio_vae'].to(device);torch.cuda.empty_cache()
                mean=torch.tensor(session['audio_vae'].config.latents_mean,device=device).view(1,-1,1)
                std=torch.tensor(session['audio_vae'].config.latents_std,device=device).view(1,-1,1)
                sound=session['audio_vae'].decode(al*std+mean,return_dict=False)[0].float().permute(1,0,2)[0].clamp(-1,1).cpu()
                wav=out/'AUDIO.wav'
                with wave.open(str(wav),'wb') as f:
                    f.setnchannels(2);f.setsampwidth(2);f.setframerate(32000)
                    f.writeframes((sound.T*32767).round().short().contiguous().numpy().tobytes())
                import imageio_ffmpeg
                mux=out/'MUX.mp4'
                subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-nostdin','-y',
                    '-i',str(video_path),'-i',str(wav),'-c:v','copy','-c:a','aac','-threads','2',
                    '-af','apad','-t',str(writer.frames/24),str(mux)],check=True,timeout=180)
                os.replace(mux,video_path)
                artifact=writer.artifact();artifact['sha256']=hashlib.sha256(video_path.read_bytes()).hexdigest()
                report=dict(status='complete_program_generated',artifact=artifact,audio=str(wav),
                    actual_frames=writer.frames,actual_duration_sec=writer.frames/24,
                    requested_duration_sec=request['requested_duration_sec'],
                    load_seconds=session['load_seconds'],audio_decode_and_mux_seconds=time.monotonic()-audio_started,
                    total_worker_wall_seconds=time.monotonic()-started,records=records,
                    timing_origin='main entry after Python imports; official job receipt includes launch/import overhead separately',
                    first_video_frames_seconds=first_video_frames_seconds,first_playable_AV_chunk_seconds=None,
                    all_chunks_NFE50_clean_commit_once=True,old_history_queries_recomputed=0,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    source_manifest=source,comparison_track=request['comparison_track'],
                    checkpoint=request['weights_spec'].get('checkpoint'),formal_memory_quality_claim=False,
                    checkpoint_sha256=request['weights_spec'].get('checkpoint_sha256'),
                    attention_contract=dispatch['attention_contract'],effective_strategy=dispatch['strategy'],
                    effective_text_reads_history=dispatch['effective_text_reads_history'],request_sha256=request_sha,
                    quality_scored=False)
                (out/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
            dist.barrier()
            peaks=[None]*dist.get_world_size()
            dist.all_gather_object(peaks,torch.cuda.max_memory_allocated()/2**30)
            if rank==0:
                report['peak_allocated_gib_by_rank']=peaks
                report['peak_allocated_gib']=max(peaks)
                (out/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:
        dist.destroy_process_group()


if __name__=='__main__':main()
