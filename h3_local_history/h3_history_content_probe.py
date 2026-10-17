"""Paired history-content replay using one frozen prefix and target noise.

Two FSDP peers run two rounds in one model session. Original/recent groups keep
all rows/positions; excluded projected state content becomes zero. No training,
source-media reading, reference-image condition or default sampler change.
"""
import argparse,datetime,hashlib,json,os,socket,subprocess,time,wave
from pathlib import Path
import imageio_ffmpeg
import torch
import torch.distributed as dist
from train_h3_v2_local_history_ddp import bootstrap
bootstrap()
from h3_fixed_chunk_contract import fixed_matched_batch
from h3_fixed_stream_decode import FixedChunkVideoDecoder
from h3_frame_writer import FrameWriter
from h3_generated_video_history import GeneratedVideoHistory
from h3_shared_loader import load_session
from h3_variable_sampling_case import decoder_chunks
from h3_window_noise_contract import combine_noise,joint_row
from h3_history_content_contract import POLICIES,content_mask,mask_projected,choose_past
from sample_demo import unpack_video


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def tensor_sha(value):return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def run_policy(root,out,meta,session,policy,device):
    out.mkdir(parents=True,exist_ok=True)
    case,source,first=meta['case'],meta['source'],meta['first_replayed_chunk'];row=joint_row(case['chunks'][first:])
    refs=[torch.load(root/f'REFERENCE_{i:03d}.pt',map_location='cpu',weights_only=True) for i in range(first,first+2)]
    video_noise,audio_noise=combine_noise(refs,case['chunks'][first:])
    prefix=torch.load(root/'PREFIX.pt',map_location='cpu',weights_only=True);start=row['video_latent_start'];aa=row['audio_latent_start']
    if prefix['completed_NN_chunk']!=first-1 or prefix['generated_ids']!=list(range(start)):raise ValueError('Incomplete frozen prefix')
    video,audio=prefix['video'].clone(),prefix['audio'].clone();video[:,start:]=0;audio[:,:,aa:]=0
    history=GeneratedVideoHistory(source);history.put(0,video[:,:start]);history.evict(start)
    ids,past=history.select(start)
    if ids!=meta['history_ids'] or any(i>=start for i in ids):raise ValueError('Retained history changed or contains future')
    donor=torch.load(root/'DONOR_PAST.pt',map_location='cpu',weights_only=True)
    if donor['donor_ids']!=meta['donor']['latent_ids'] or max(donor['donor_ids'])>=meta['donor']['available_before_latent']:
        raise ValueError('Wrong-history control reads donor future')
    selected=choose_past(past,donor['video'],policy)
    prompt=torch.load(root/f'PROMPT_{first:03d}.pt',map_location='cpu',weights_only=True)
    generator=torch.Generator(device=device).manual_seed(case['seed']);mask=torch.ones(source['audio_latent_count'],dtype=torch.bool)
    batch=fixed_matched_batch(row,source,video,audio,prompt['prompt_embeds'],prompt['text_token_tags'],mask,generator,device,
                              retained_ids=ids,retained_video=selected)
    batch.update(sampling=True,zero_history_state=False);inputs=batch['inputs'];model=session['model']
    for processor in session['processors']:
        processor.text_policy='text_full';processor.prefix_rows=batch['prefix_rows']
        processor.text_rows=int(inputs['text_indices'].numel());processor.prefix_reads_current=not ids
    rows_per_latent=batch['history_rows'].shape[1]//len(ids)
    row_mask=content_mask(ids,meta['original_ids'],meta['recent_ids'],policy,rows_per_latent,device)
    seen=dict(calls=0,first_projected_content_sha256=None,excluded_rows=0)
    def intervene(module,args,output):
        value=mask_projected(output,row_mask);seen['calls']+=1
        if row_mask is not None and torch.count_nonzero(value.masked_select((~row_mask).expand_as(value))).item()!=0:
            raise ValueError('Excluded projected state rows remain nonzero')
        if seen['first_projected_content_sha256'] is None:
            seen['first_projected_content_sha256']=tensor_sha(value)
            seen['excluded_rows']=0 if row_mask is None else int((~row_mask).sum())
        return value
    hook=model.projection.register_forward_hook(intervene)
    positions=inputs['position_ids'];position_hashes={k:tensor_sha(positions[inputs[k+'_indices']].float()) for k in ['text','state','audio','video']}
    xv,xa=video_noise.to(device).clone(),audio_noise.to(device).clone()
    if xv.shape!=inputs['hidden_states'].shape or xa.shape!=inputs['audio_hidden_states'].shape:raise ValueError('Frozen target noise geometry differs')
    before_prefix_video=tensor_sha(video[:,:start]);before_prefix_audio=tensor_sha(audio[:,:,:aa]);nfe=meta['NFE']
    config=dict(policy=policy,history_ids=ids,original_ids=meta['original_ids'],recent_ids=meta['recent_ids'],
        generation_row=row,decode_rows=case['chunks'],weights_spec=meta['weights_spec'],NFE=nfe,CFG=1.,
        input_manifest_sha256=file_sha(root/'INPUT.json'),history_raw_sha256=tensor_sha(selected),
        original_raw_history_sha256=tensor_sha(past),position_hashes_fp32=position_hashes,
        video_noise_sha256=tensor_sha(video_noise),audio_noise_sha256=tensor_sha(audio_noise),
        state_tokens_removed=False,mask_stage='after learned history projection; retained rows and positions unchanged',
        source_media_read=False,source_audio_encode=False,audio_history_used=False,
        audio_future_context_allowed=False,audio_encoder_lookahead_sec=0.,audio_decode='offline complete generated clip',
        formal_memory_quality_claim=False,generated_clock_only=True)
    atomic(out/'RESOLVED_CONFIG.json',config);torch.cuda.reset_peak_memory_stats();dist.barrier();started=time.monotonic()
    try:
        with torch.inference_mode():
            for i in range(nfe):
                u,unext=1-i/nfe,1-(i+1)/nfe;tv,ta=1-12*u/(1+11*u),1-3*u/(1+2*u)
                vn,an=1-12*unext/(1+11*unext),1-3*unext/(1+2*unext)
                rt=torch.full((len(positions),),tv,device=device);rt[inputs['state_indices']]=1.;rt[inputs['audio_indices']]=ta
                inputs['timestep'],inputs['timestep_indices']=torch.unique(rt,sorted=True,return_inverse=True)
                inputs['hidden_states'],inputs['audio_hidden_states']=xv,xa;pv,pa=model(batch)
                xv,xa=xv+(vn-tv)*pv.float(),xa+(an-ta)*pa.float()
                if i%10==0 or i==nfe-1:
                    torch.cuda.synchronize();print(json.dumps(dict(event='history_content_NFE',policy=policy,NFE=i+1,seconds=time.monotonic()-started)),flush=True)
            torch.cuda.synchronize();dit_seconds=time.monotonic()-started
            a,b=row['video_latent_start'],row['video_latent_end'];ae=row['audio_latent_end']
            chunk=unpack_video(xv,b-a,48,84).to(torch.bfloat16)[0].cpu()
            new_audio=xa[0].reshape(2,ae-aa,32).permute(0,2,1).to(torch.bfloat16).cpu()
            if not torch.isfinite(chunk).all() or not torch.isfinite(new_audio).all():raise ValueError('Nonfinite generated suffix')
            video[:,a:b],audio[:,:,aa:ae]=chunk,new_audio
            if tensor_sha(video[:,:start])!=before_prefix_video or tensor_sha(audio[:,:,:aa])!=before_prefix_audio:
                raise ValueError('Replay modified committed prefix')
            generated=out/'GENERATED.pt';torch.save(dict(video=video,audio=audio,positions_fp32={k:positions[inputs[k+'_indices']].float().cpu() for k in ['text','state','audio','video']}),generated)
            reference=torch.load(root/'CONTROL10_GENERATED.pt',map_location='cpu',weights_only=True)
            differences={k:float((reference[k].float()-v.float()).abs().max()) for k,v in [('video',video),('audio',audio)]}
            if seen['calls']!=nfe:raise ValueError('History projection hook did not execute every NFE')
            vae=session['vae'];vae.to(device);decoder=FixedChunkVideoDecoder(vae,device);previous=None;path=out/'video.mp4'
            def emit(writer,value):
                if value is not None:
                    if not torch.isfinite(value).all():raise ValueError('Nonfinite video decode')
                    writer.append(value.cpu())
            with FrameWriter(path,case['nominal_output_fps']) as writer:
                for original in case['chunks']:
                    if previous is not None and previous!=original['parent_window_id']:emit(writer,decoder.flush())
                    left,right=original['video_latent_start'],original['video_latent_end']
                    for part in decoder_chunks(video[None,:,left:right].to(device),left):emit(writer,decoder.push(part))
                    previous=original['parent_window_id']
                emit(writer,decoder.flush());writer.close();artifact=writer.artifact();frames=writer.frames
            if frames!=source['valid_output_frames']:raise ValueError('Content intervention changed output frame count')
            vae.cpu();del decoder;torch.cuda.empty_cache();avae=session['audio_vae'];avae.to(device)
            mean=torch.tensor(avae.config.latents_mean,device=device).view(1,-1,1);std=torch.tensor(avae.config.latents_std,device=device).view(1,-1,1)
            decoded=avae.decode(audio.to(device)*std+mean,return_dict=False)[0].float()
            if not torch.isfinite(decoded).all():raise ValueError('Nonfinite generated audio decode')
            aw=decoded.permute(1,0,2)[0].clamp(-1,1).cpu();wav=out/'audio.wav'
            with wave.open(str(wav),'wb') as stream:
                stream.setnchannels(2);stream.setsampwidth(2);stream.setframerate(32000);stream.writeframes((aw.T*32767).round().short().contiguous().numpy().tobytes())
            mux=out/'mux.mp4'
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-nostdin','-y','-i',str(path),'-i',str(wav),'-c:v','copy','-c:a','aac','-af','apad','-t',str(frames/case['nominal_output_fps']),str(mux)],check=True,timeout=120)
            os.replace(mux,path);artifact.update(sha256=file_sha(path),audio_muxed=True);avae.cpu();torch.cuda.empty_cache()
        result=dict(status='complete',policy=policy,artifact=artifact,generated=str(generated),generated_sha256=file_sha(generated),
            config=config,projection_intervention=seen,unintervened_control_tensor_maxabs=differences,
            prefix_video_and_audio_preserved=True,DiT_seconds=dit_seconds,wall_sec=time.monotonic()-started,
            checkpoint_sha256=meta['weights_spec']['checkpoint_sha256'],peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            hardware_host=socket.gethostname(),hardware_gpu=torch.cuda.get_device_name(device),hardware_total_gib=torch.cuda.get_device_properties(device).total_memory/2**30,
            input_manifest_sha256=file_sha(root/'INPUT.json'),formal_memory_quality_claim=False,training_or_default_policy_changed=False)
        atomic(out/'RESULT.json',result);return result
    finally:hook.remove()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--out',required=True);args=parser.parse_args()
    root,out=Path(args.input),Path(args.out);out.mkdir(parents=True,exist_ok=True);meta=json.loads((root/'INPUT.json').read_text())
    rank,local,world=[int(os.environ[k]) for k in ['RANK','LOCAL_RANK','WORLD_SIZE']]
    if world!=2:raise ValueError('Exactly two FSDP peers required')
    for name,digest in meta['input_file_sha256'].items():
        if file_sha(root/name)!=digest:raise ValueError('Frozen content-probe input changed: '+name)
    if meta['NFE']!=50 or tuple(meta['policies'])!=POLICIES:raise ValueError('Unexpected diagnostic protocol')
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(2)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=60))
    try:
        session=load_session(meta['weights_spec'],device)
        for policies in [POLICIES[:2],POLICIES[2:]]:
            run_policy(root,out/policies[rank],meta,session,policies[rank],device);dist.barrier()
        if rank==0:
            results=[json.loads((out/p/'RESULT.json').read_text()) for p in POLICIES]
            fields=['history_ids','generation_row','position_hashes_fp32','video_noise_sha256','audio_noise_sha256','NFE','CFG']
            for value in results:
                if any(value['config'][k]!=results[0]['config'][k] for k in fields):raise ValueError('Content controls changed common target inputs')
            full=results[0]
            atomic(out/'CONTROL_RESULT.json',dict(all4_same_prefix_positions_noise_NFE_geometry=True,
                full_exactly_reproduces_prior10_window=all(v==0 for v in full['unintervened_control_tensor_maxabs'].values()),
                policies=list(POLICIES),results=[str(out/p/'RESULT.json') for p in POLICIES],
                state_row_removal_claim=False,semantic_review_pending=True,formal_memory_quality_claim=False))
    finally:dist.destroy_process_group()
if __name__=='__main__':main()
