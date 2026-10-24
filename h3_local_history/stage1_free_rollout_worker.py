"""Stage1 long causal free-rollout health worker.

Repeats the declared 7+5+5+5 chunk contract over an explicit 60-second
generated clock. It writes every generated chunk and no optimizer state.
"""
import argparse, datetime, json, os, sys, time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--request',required=True);a=ap.parse_args();req=json.loads(Path(a.request).read_text())
    src=Path(req['stage1_source_dir']);sys.path.insert(0,str(src))
    from train_h3_worldattention_dmd_fsdp2 import CHUNKS, build_student, generation_step
    from train_h3_worldattention_ode_fsdp2 import chunk_contract
    from h3_worldattention_causal import VideoKVBank
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=datetime.timedelta(hours=3),device_id=device);mesh=init_device_mesh('cuda',(world,));out=Path(os.environ['INFER_OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    try:
        if world!=8:raise ValueError('Stage1 free rollout requires eight ranks')
        prompt=json.loads(Path(req['prompt_pool']).read_text().splitlines()[0]);emb=torch.load(prompt['cached_embedding_path_candidate'],map_location='cpu',weights_only=True)
        cycles=int(req.get('cycles',22))
        if cycles < 1:
            raise ValueError('cycles must be positive')
        # The native prefix is 7+5+5+5.  Once that prefix has been emitted,
        # every continuation cycle adds four 5-latent blocks; repeating a new
        # 7-latent prefix at an absolute offset (e.g. 29) violates H3's
        # 2+5*n boundary contract.
        first_chunks=list(CHUNKS)
        continuation_chunks=[(0,5),(5,10),(10,15),(15,20)]
        total_latents=22+20*(cycles-1)
        # Native H3 time is 5 initial pixel frames followed by 17 frames for
        # each complete 5-latent continuation group.  The previous draft
        # reported latent_count*17/24, which overstated a 20-cycle rollout by
        # almost 5x.  Keep the same absolute boundary formula used by
        # chunk_contract for the actual audio/video clock.
        pixel_frames=5+17*((total_latents-2)//5)
        audio_end=round(pixel_frames*40/24)
        video=torch.zeros(24,total_latents,48,84,dtype=torch.bfloat16);audio=torch.zeros(2,32,audio_end,dtype=torch.bfloat16)
        model,processors=build_student(req['base_source'],req['export'],mesh,device,'student',True,8192,4096,rank)
        bank=VideoKVBank(local_frames=12,sink_frames=3);generator=torch.Generator(device=device).manual_seed(req['seed']);records=[];chunk_number=0
        for cycle in range(cycles):
            schedule=first_chunks if cycle == 0 else continuation_chunks
            offset=0 if cycle == 0 else 22+20*(cycle-1)
            for local_index,(local_start,local_end) in enumerate(schedule):
                start=offset+local_start;end=offset+local_end
                a1=audio_end if end==total_latents else round((5+17*((end-2)//5))*40/24)
                batch=chunk_contract(video,audio,emb['prompt_embeds'],emb['text_token_tags'],generator,device,start,end,a1)
                tick=time.monotonic();generated,generated_audio,initial=generation_step(model,processors,batch,chunk_number,start,end,bank,generator,req['nfe'],device,False)
                torch.cuda.synchronize();elapsed=time.monotonic()-tick
                finite=bool(torch.isfinite(generated).all() and torch.isfinite(generated_audio).all())
                rec=dict(chunk=chunk_number,cycle=cycle,local_chunk=local_index,start_latent=start,end_latent=end,latent_mean=float(generated.float().abs().mean()),latent_std=float(generated.float().std()),latent_max=float(generated.float().abs().max()),finite=finite,wall_seconds=elapsed,kv_bytes=bank.bytes)
                if rank == 0:
                    records.append(rec)
                    torch.save(dict(video=generated.cpu(),audio=generated_audio.cpu(),initial_video=initial[0].cpu(),initial_audio=initial[1].cpu(),record=rec),out/f'chunk{chunk_number:03d}.pt')
                if not finite:raise RuntimeError('Nonfinite Stage1 rollout output')
                chunk_number+=1
        if rank==0:(out/'RESULT.json').write_text(json.dumps(dict(status='stage1_step8_60s_free_rollout_complete',checkpoint=req['export'],nfe=req['nfe'],cycles=cycles,total_latents=total_latents,pixel_frames=pixel_frames,audio_latents=audio_end,requested_seconds=req.get('requested_seconds'),actual_clock_seconds=pixel_frames/24,chunks=records,history_contract='causal_local12_sink3_readonly_video_KV'),indent=2)+'\n')
        dist.barrier()
    finally:dist.destroy_process_group()
if __name__=='__main__':main()
