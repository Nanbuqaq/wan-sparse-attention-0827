"""Bounded three-way ABC history diagnostic.

Modes:
  fixed      : immutable committed KV, clean commit once per block.
  recompute  : rebuild the same-rule clean history once before each block, then
               keep it fixed during that block's 50-NFE denoising.
  reset      : immutable KV, cleared only when the prompt/shot changes.

This is a diagnostic worker. It is not a production benchmark and the
recompute lane is explicitly allowed to re-encode history once per block.
"""
import argparse, datetime, hashlib, json, os, subprocess, time, wave
from pathlib import Path
import imageio_ffmpeg
import torch
import torch.distributed as dist
from train_h3_v2_local_history_ddp import bootstrap
bootstrap()
from h3_shared_loader import load_session
from h3_shared_stage1 import shared_modules, TransformerCallProxy
from h3_shared_kv_heads import require_same_request
from h3_buffered_video_decode import BufferedVideoDecoder
from h3_frame_writer import FrameWriter
from sample_demo import unpack_video


def tensor_sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def block_template(row, prompt, origin, device='cpu'):
    start = row['local_latent_start']; frame0 = row.get('shot_frame_start', 0)
    local_frames = [17*(i//5)+(0,1,5,9,13)[i%5] for i in range(start, start+5)]
    positions = torch.empty((5,24,42,3))
    positions[:,:,:,0] = torch.tensor([origin+40*(frame0+f)/24 for f in local_frames])[:,None,None]
    positions[:,:,:,1] = torch.arange(24)[None,:,None]
    positions[:,:,:,2] = torch.arange(42)[None,None,:]
    a0,a1 = row['audio_start'], row['audio_end']
    audio_pos = torch.zeros(2,a1-a0,3)
    audio_pos[:,:,0] = origin + torch.arange(a0,a1)[None]
    audio_pos[:,:,1] = torch.arange(2)[:,None]
    return dict(block=row['block'], video=torch.zeros(5040,96), audio=torch.zeros(2*(a1-a0),32),
                prompt=prompt['prompt_embeds'], video_pos=positions.reshape(-1,3),
                audio_pos=audio_pos.reshape(-1,3), audio_mask=torch.ones(2*(a1-a0),dtype=torch.bool),
                source_start=row['source_start'], source_end=row['source_end'], kind='fixed5',
                past=tuple(row.get('past', ())), target=True)


def generate_current(proxy, code, block, generator, device, strategy, nfe, bank, recompute_bank=None):
    core=code['h3_fixed_kv_stage1']; data=code['h3_stage1_abc_data']; runtime=code['h3_fixed_kv_runtime']
    batch=data.pack([block], device, generator, strategy=strategy, include_clean=False)
    inputs=batch['inputs']; video=torch.randn(inputs['hidden_states'].shape,device=device,generator=generator)
    audio=torch.randn(inputs['audio_hidden_states'].shape,device=device,generator=generator)
    context=core.Context(batch['context'].splits, strategy=='B', bank.snapshot())
    for step in range(nfe):
        if recompute_bank is not None:
            bank_for_step=recompute_bank()
            context=core.Context(batch['context'].splits, strategy=='B', bank_for_step.snapshot())
        u=1-step/nfe; un=1-(step+1)/nfe
        tv,ta=1-12*u/(1+11*u),1-3*u/(1+2*u)
        nv,na=1-12*un/(1+11*un),1-3*un/(1+2*un)
        clock=torch.ones(len(inputs['position_ids']),device=device)
        clock[inputs['video_indices']]=tv; clock[inputs['audio_indices']]=ta
        inputs['timestep'],inputs['timestep_indices']=torch.unique(clock,sorted=True,return_inverse=True)
        inputs['hidden_states'],inputs['audio_hidden_states']=video,audio
        pv,pa=core.forward(proxy,inputs,context)
        video=video+(nv-tv)*pv.float(); audio=audio+(na-ta)*pa.float()
    if not bool(torch.isfinite(video).all() and torch.isfinite(audio).all()):
        raise RuntimeError('nonfinite generated chunk')
    return video,audio


def rebuild_history(code, proxy, history, current_start, device, strategy, generator):
    core=code['h3_fixed_kv_stage1']; data=code['h3_stage1_abc_data']
    keep=[h for h in history if h['row']['source_end'] <= current_start+1e-7 and h['row']['source_start'] >= current_start-8.-1e-7]
    bank=core.VideoBank(detach=True)
    for h in keep:
        row=dict(h['block'], video=h['video'], audio=h['audio'], past=tuple(x['row']['block'] for x in keep if x['row']['source_end'] <= h['row']['source_start']+1e-7), target=False)
        batch=data.pack([row],device,generator,strategy=strategy,clean_only=True)
        ctx=core.Context(batch['context'].splits,strategy=='B',bank.snapshot(),bank)
        core.forward(proxy,batch['inputs'],ctx)
    return bank,keep


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--request',required=True); ap.add_argument('--mode',choices=['fixed','recompute','reset'],required=True); args=ap.parse_args()
    request=json.loads(Path(args.request).read_text()); mode=args.mode
    rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK']); world=int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local); device=torch.device('cuda',local); torch.set_num_threads(2)
    dist.init_process_group('nccl',timeout=datetime.timedelta(hours=3),device_id=device)
    out=Path(os.environ['INFER_OUTPUT_DIR'])/mode; out.mkdir(parents=True,exist_ok=True)
    try:
        if world!=8: raise ValueError('ABC diagnostic requires eight ranks')
        req_hash=hashlib.sha256(Path(args.request).read_bytes()).hexdigest(); require_same_request(req_hash)
        code,source=shared_modules(); session=load_session(request['weights_spec'],device)
        proxy=TransformerCallProxy(session['model']); core=code['h3_fixed_kv_stage1']; runtime=code['h3_fixed_kv_runtime']
        prompts=[torch.load(p,map_location='cpu',weights_only=True) for p in request['prompt_paths']]
        for item,text in zip(prompts,request['prompts']):
            if item['prompt']!=text or not bool((item['text_token_tags']==1).all()): raise ValueError('prompt mismatch')
        origin=prompts[0]['prompt_embeds'].shape[0]; rows=request['blocks']; generator=torch.Generator(device=device).manual_seed(request['seed'])
        stream=runtime.FixedKVStream(proxy,request.get('strategy','A'))
        history=[]; records=[]; audio_pieces=[]; decoder=BufferedVideoDecoder(session['vae'],device) if rank==0 else None
        video_path=out/'VIDEO.mp4'; writer=FrameWriter(video_path,24,encoder_threads=2) if rank==0 else None
        if rank==0: (out/'RESOLVED_CONFIG.json').write_text(json.dumps(dict(request=request,mode=mode,source_manifest=source,weights_metadata=session['training_metadata'],history_recomputation_allowed=(mode=='recompute'),diagnostic_only=True),indent=2)+'\n')
        with torch.inference_mode():
            for row0 in rows:
                row=dict(row0,source_start=row0['start_frame']/24,source_end=row0['end_frame']/24)
                prompt=prompts[row['prompt_index']]; row['past']=tuple(h['row']['block'] for h in history if h['row']['source_end']<=row['source_start']+1e-7 and h['row']['source_start']>=row['source_start']-8.-1e-7)
                if mode=='reset' and history and history[-1]['row']['prompt_index']!=row['prompt_index']:
                    stream.reset(); history=[]; row['past']=()
                block=block_template(row,prompt,origin)
                before=generator.get_state().cpu(); tick=time.monotonic(); recompute_count=0
                if mode=='fixed':
                    video,audio,stats=stream.generate(block,generator,device,50)
                elif mode=='reset':
                    video,audio,stats=stream.generate(block,generator,device,50)
                else:
                    rebuilt,keep=rebuild_history(code,proxy,history,row['source_start'],device,request.get('strategy','A'),generator)
                    recompute_count=len(keep)
                    video,audio=generate_current(proxy,code,block,generator,device,request.get('strategy','A'),50,rebuilt,recompute_bank=None)
                    stats=dict(block=row['block'],read_past_blocks=tuple(h['row']['block'] for h in keep),KV_bytes=rebuilt.bytes,clean_commit_forwards=0,old_video_queries_recomputed=recompute_count)
                torch.cuda.synchronize(); wall=time.monotonic()-tick
                latent=unpack_video(video,5,48,84).to(torch.bfloat16)
                sound=audio[0].reshape(2,-1,32).permute(0,2,1).to(torch.bfloat16)
                if rank==0:
                    snap=out/f'chunk{row["block"]:03d}_GENERATED.pt'; torch.save(dict(video=latent.cpu(),audio=sound.cpu(),generator_before=before,generator_after=generator.get_state().cpu(),row=row),snap)
                    decoded=decoder.push(latent)
                    if decoded is not None: writer.append(decoded)
                    audio_pieces.append(sound)
                    records.append(dict(block=row['block'],prompt_index=row['prompt_index'],history_blocks=list(stats['read_past_blocks']),history_rebuilt_blocks=recompute_count,KV_bytes=stats['KV_bytes'],sample_wall_seconds=wall,finite=True,old_history_queries_recomputed=recompute_count,seed=request['seed'],generator_before_sha256=tensor_sha(before)))
                history.append(dict(row=row,block=block,video=video[0].float().detach(),audio=audio[0].float().detach()))
                dist.barrier()
            if rank==0:
                tail=decoder.finish()
                if tail is not None: writer.append(tail)
                writer.close()
                if writer.frames != request['actual_frames']: raise ValueError(f'frame count {writer.frames} != {request["actual_frames"]}')
                al=torch.cat(audio_pieces,dim=2); av=session['audio_vae']; av.to(device)
                mean=torch.tensor(av.config.latents_mean,device=device).view(1,-1,1); std=torch.tensor(av.config.latents_std,device=device).view(1,-1,1)
                sound=av.decode(al*std+mean,return_dict=False)[0].float().permute(1,0,2)[0].clamp(-1,1).cpu(); wav=out/'AUDIO.wav'
                with wave.open(str(wav),'wb') as f:
                    f.setnchannels(2); f.setsampwidth(2); f.setframerate(32000); f.writeframes((sound.T*32767).round().short().contiguous().numpy().tobytes())
                mux=out/'MUX.mp4'; subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-nostdin','-y','-i',str(video_path),'-i',str(wav),'-c:v','copy','-c:a','aac','-threads','2','-af','apad','-t',str(writer.frames/24),str(mux)],check=True,timeout=180); os.replace(mux,video_path)
                (out/'RESULT.json').write_text(json.dumps(dict(status='complete_diagnostic',mode=mode,actual_frames=writer.frames,actual_duration_sec=writer.frames/24,records=records,old_history_queries_recomputed=sum(x['old_history_queries_recomputed'] for x in records),checkpoint=request['weights_spec']['checkpoint'],checkpoint_sha256=request['weights_spec'].get('checkpoint_sha256'),comparison_track='same_backbone_h3_engineering_diagnostic',formal_memory_quality_claim=False,quality_scored=False,source_manifest=source,video_sha256=hashlib.sha256(video_path.read_bytes()).hexdigest(),wall_seconds=time.monotonic()-tick),indent=2)+'\n')
            dist.barrier()
    finally: dist.destroy_process_group()

if __name__=='__main__': main()
