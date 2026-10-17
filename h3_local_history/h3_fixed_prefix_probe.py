"""Replay a generated suffix with identical saved prefix/noise and model weights.

Two FSDP ranks compare text_full and text_current. This is a diagnostic replay,
not a new training method or a persistent KV cache. Inputs remain private.
"""
import argparse,datetime,hashlib,json,os,subprocess,time,wave
from pathlib import Path

import torch
import torch.distributed as dist
import imageio_ffmpeg
from train_h3_v2_local_history_ddp import bootstrap
bootstrap()
from h3_shared_loader import load_session
from h3_fixed_chunk_contract import fixed_matched_batch
from h3_generated_video_history import GeneratedVideoHistory
from h3_fixed_stream_decode import FixedChunkVideoDecoder
from h3_variable_sampling_case import decoder_chunks
from h3_frame_writer import FrameWriter
from sample_demo import unpack_video

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(16*1024*1024),b''):h.update(data)
    return h.hexdigest()

def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--out',required=True);args=parser.parse_args()
    root=Path(args.input);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((root/'INPUT.json').read_text());rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=2:raise ValueError('This paired replay requires2 ranks with identical generation geometry')
    policy=['text_full','text_current'][rank];torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(2)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=60))
    try:
        for rel,expected in meta['input_file_sha256'].items():
            if file_sha(root/rel)!=expected:raise ValueError('Frozen probe input changed: '+rel)
        prefix=torch.load(root/'PREFIX.pt',map_location='cpu',weights_only=True)
        case=meta['case'];source=meta['source'];first=meta['first_replayed_chunk']
        if prefix['completed_NN_chunk']!=first-1:raise ValueError('Saved prefix must end exactly before replayed suffix')
        start=case['chunks'][first]['video_latent_start']
        if prefix['generated_ids']!=list(range(start)):raise ValueError('Probe prefix is not a complete generated prefix')
        session=load_session(meta['weights_spec'],device);model=session['model'];processors=session['processors']
        for p in processors:p.text_policy=policy
        video=prefix['video'].clone();audio=prefix['audio'].clone();video[:,start:]=0
        audio_start=case['chunks'][first]['audio_latent_start'];audio[:,:,audio_start:]=0
        history=GeneratedVideoHistory(source);history.put(0,video[:,:start]);history.evict(start)
        generator=torch.Generator(device=device).manual_seed(case['seed']);mask=torch.ones(source['audio_latent_count'],dtype=torch.bool)
        records=[];torch.cuda.reset_peak_memory_stats();dist.barrier();begin=time.monotonic()
        with torch.inference_mode():
            for number in range(first,len(case['chunks'])):
                row=case['chunks'][number];reference=torch.load(root/f'REFERENCE_{number:03d}.pt',map_location='cpu',weights_only=True)
                prompt=torch.load(root/f'PROMPT_{number:03d}.pt',map_location='cpu',weights_only=True)
                if reference['current_chunk']!=row:raise ValueError('Reference chunk geometry differs')
                chosen,past=history.select(row['video_latent_start'])
                if any(i>=row['video_latent_start'] for i in chosen):raise ValueError('Future frame in replay history')
                history_hash=tensor_sha(past) if past is not None else None
                batch=fixed_matched_batch(row,source,video,audio,prompt['prompt_embeds'],prompt['text_token_tags'],mask,generator,device,
                                          retained_ids=chosen,retained_video=past)
                batch.update(sampling=True,zero_history_state=False);inputs=batch['inputs']
                for p in processors:
                    p.prefix_rows=batch['prefix_rows'];p.text_rows=int(inputs['text_indices'].numel());p.prefix_reads_current=not chosen
                xv=reference['initial_video_noise'].to(device);xa=reference['initial_audio_noise'].to(device)
                if xv.shape!=inputs['hidden_states'].shape or xa.shape!=inputs['audio_hidden_states'].shape:raise ValueError('Saved noise geometry differs')
                if tensor_sha(xv)!=reference['current_video_noise_sha256'] or tensor_sha(xa)!=reference['current_audio_noise_sha256']:raise ValueError('Saved noise hash differs')
                nfe=reference['nfe'];positions_sha=tensor_sha(inputs['position_ids']);tick=time.monotonic()
                for i in range(nfe):
                    u=1-i/nfe;unext=1-(i+1)/nfe;tv=1-12*u/(1+11*u);ta=1-3*u/(1+2*u)
                    vn=1-12*unext/(1+11*unext);an=1-3*unext/(1+2*unext)
                    rt=torch.full((len(inputs['position_ids']),),tv,device=device);rt[inputs['state_indices']]=1.;rt[inputs['audio_indices']]=ta
                    inputs['timestep'],inputs['timestep_indices']=torch.unique(rt,sorted=True,return_inverse=True)
                    inputs['hidden_states']=xv;inputs['audio_hidden_states']=xa;pv,pa=model(batch)
                    xv=xv+(vn-tv)*pv.float();xa=xa+(an-ta)*pa.float()
                    if i in [0,nfe-1]:
                        torch.cuda.synchronize();print(json.dumps(dict(event='probe_NFE',policy=policy,chunk=number,NFE=i+1,seconds=time.monotonic()-tick)),flush=True)
                a,b=row['video_latent_start'],row['video_latent_end'];aa,ae=row['audio_latent_start'],row['audio_latent_end']
                chunk=unpack_video(xv,b-a,48,84).to(torch.bfloat16)[0].cpu();generated_audio=xa[0].reshape(2,ae-aa,32).permute(0,2,1).to(torch.bfloat16).cpu()
                if not torch.isfinite(chunk).all() or not torch.isfinite(generated_audio).all():raise ValueError('Nonfinite replay output')
                video[:,a:b]=chunk;audio[:,:,aa:ae]=generated_audio;history.put(a,chunk);history.evict(b)
                error=(chunk.float()-reference['video'][:,a:b].float()).abs();audio_error=(generated_audio.float()-reference['audio'][:,:,aa:ae].float()).abs()
                records.append(dict(chunk=number,policy=policy,history_ids=chosen,history_sha256=history_hash,positions_sha256=positions_sha,
                                    initial_video_noise_sha256=reference['current_video_noise_sha256'],initial_audio_noise_sha256=reference['current_audio_noise_sha256'],
                                    reference_video_max_abs=float(error.max()),reference_audio_max_abs=float(audio_error.max()),NFE=nfe,DiT_seconds=time.monotonic()-tick,
                                    generated_video_sha256=tensor_sha(chunk),generated_audio_sha256=tensor_sha(generated_audio)))
            generated=out/(policy+'_GENERATED.pt');torch.save(dict(video=video,audio=audio,records=records),generated)
            vae=session['vae'];vae.to(device);decoder=FixedChunkVideoDecoder(vae,device);previous=None
            path=out/(policy+'.mp4')
            with FrameWriter(path,case['nominal_output_fps']) as writer:
                for row in case['chunks']:
                    if previous is not None and previous!=row['parent_window_id']:
                        tail=decoder.flush()
                        if tail is not None:writer.append(tail.cpu())
                    a,b=row['video_latent_start'],row['video_latent_end']
                    for part in decoder_chunks(video[None,:,a:b].to(device),a):writer.append(decoder.push(part).cpu())
                    previous=row['parent_window_id']
                tail=decoder.flush()
                if tail is not None:writer.append(tail.cpu())
                writer.close();artifact=writer.artifact();frames=writer.frames
            if frames!=source['valid_output_frames']:raise ValueError('Replay output frame count differs')
            vae.cpu();torch.cuda.empty_cache();avae=session['audio_vae'];avae.to(device)
            mean=torch.tensor(avae.config.latents_mean,device=device).view(1,-1,1);std=torch.tensor(avae.config.latents_std,device=device).view(1,-1,1)
            aw=avae.decode(audio.to(device)*std+mean,return_dict=False)[0].float().permute(1,0,2)[0].clamp(-1,1).cpu();wav=path.with_suffix('.wav')
            with wave.open(str(wav),'wb') as stream:
                stream.setnchannels(2);stream.setsampwidth(2);stream.setframerate(32000);stream.writeframes((aw.T*32767).round().short().contiguous().numpy().tobytes())
            mux=path.with_suffix('.mux.mp4')
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-nostdin','-y','-i',str(path),'-i',str(wav),'-c:v','copy','-c:a','aac','-af','apad',
                            '-t',str(frames/case['nominal_output_fps']),str(mux)],check=True,timeout=120);os.replace(mux,path)
            artifact.update(sha256=file_sha(path),audio_muxed=True)
        result=dict(policy=policy,status='complete',weights_spec=meta['weights_spec'],records=records,artifact=artifact,generated=str(generated),
                    first_replayed_chunk=first,prefix_sha256=meta['input_file_sha256']['PREFIX.pt'],input_manifest_sha256=file_sha(root/'INPUT.json'),
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,wall_sec=time.monotonic()-begin,host=__import__('socket').gethostname(),GPU=torch.cuda.get_device_name(device),
                    training_changed=False,default_policy_changed=False,formal_memory_quality_claim=False,offline_audio_decode_only=True)
        (out/(policy+'_RESULT.json')).write_text(json.dumps(result,indent=2)+'\n');dist.barrier()
        if rank==0:
            values=[json.loads((out/(p+'_RESULT.json')).read_text()) for p in ['text_full','text_current']]
            for a,b in zip(values[0]['records'],values[1]['records']):
                for key in ['chunk','positions_sha256','initial_video_noise_sha256','initial_audio_noise_sha256','NFE']:
                    if a[key]!=b[key]:raise ValueError('Paired replay inputs differ: '+key)
            if values[0]['records'][0]['history_sha256']!=values[1]['records'][0]['history_sha256']:raise ValueError('First replay history differs')
            exact=all(r['reference_video_max_abs']==0 and r['reference_audio_max_abs']==0 for r in values[0]['records'])
            (out/'CONTROL_RESULT.json').write_text(json.dumps(dict(same_prefix_and_first_history=True,same_noise_positions_weights=True,
                   text_full_exactly_reproduces_saved_suffix=exact,semantic_review_pending=True),indent=2)+'\n')
    finally:dist.destroy_process_group()

if __name__=='__main__':main()
