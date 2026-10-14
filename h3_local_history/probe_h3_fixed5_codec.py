#!/usr/bin/env python3
"""Real-resolution native decoder equality and hard-cut boundary audit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import torch
from train_h3_v2_local_history_ddp import bootstrap
from h3_fixed_stream_decode import FixedChunkVideoDecoder


def video_file(path,decoded,fps):
    import imageio_ffmpeg
    mean=torch.tensor((.485,.456,.406),device=decoded.device).view(1,3,1,1,1)
    std=torch.tensor((.229,.224,.225),device=decoded.device).view(1,3,1,1,1)
    pixels=((decoded.float()*std+mean).clamp(0,1)*255).round().byte()[0].permute(1,2,3,0).cpu().contiguous()
    h,w=pixels.shape[1:3]
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-nostdin','-y','-f','rawvideo','-pixel_format','rgb24',
        '-video_size',f'{w}x{h}','-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264',
        '-preset','fast','-crf','18','-pix_fmt','yuv420p',str(path)],input=pixels.numpy().tobytes(),check=True)
    return dict(path=str(path),frames=len(pixels),fps=fps,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    bootstrap()
    from diffusers import AutoencoderKLMiniMaxH3
    ap=argparse.ArgumentParser();ap.add_argument('--base-source',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--rank',type=int,default=0)
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda',args.rank);torch.cuda.set_device(device);torch.set_num_threads(2)
    started=time.monotonic();vae=AutoencoderKLMiniMaxH3.from_pretrained(args.base_source,subfolder='vae').to(device).eval()
    decoder=FixedChunkVideoDecoder(vae,device)
    source=Path('/kaimm-distill/zhouhe08/h3_train_runs/vdn_h3_full_20261010/h3_temporal_aligned_v2_media_20261010/shards/rank0/sources/6058965007/video.pt')
    normalized=torch.load(source,map_location='cpu',weights_only=True)[None].to(device)
    with torch.inference_mode():
        # Source startup and two same-shot continuations, on the actual 48x84 latent canvas.
        prefix=normalized[:,:,:17]
        whole=decoder.decode(prefix)
        pieces=[decoder.push(prefix[:,:,:7]),decoder.push(prefix[:,:,7:12]),decoder.push(prefix[:,:,12:17]),decoder.flush()]
        streamed=torch.cat(pieces,dim=2)
        error=float((whole.float()-streamed.float()).abs().max())
        if whole.shape!=streamed.shape or error>1e-5:raise RuntimeError(f'Native stream mismatch: {error}')
        # Start a new shot from its own two repeated latent prefix; no old shot is blended.
        new_shot=normalized[:,:,47:62]
        reference=decoder.decode(torch.cat((new_shot[:,:,:1].repeat(1,1,2,1,1),new_shot),dim=2))[:,:,5:]
        parts=[decoder.push(new_shot[:,:,:5]),decoder.push(new_shot[:,:,5:10]),decoder.push(new_shot[:,:,10:15]),decoder.flush()]
        new_stream=torch.cat(parts,dim=2)
        cut_error=float((reference.float()-new_stream.float()).abs().max())
        if reference.shape!=new_stream.shape or cut_error>1e-5:raise RuntimeError(f'New-shot stream mismatch: {cut_error}')
        if decoder.pending is not None or decoder.previous_two is not None:raise RuntimeError('Shot flush retained old-scene decoder state.')
        artifacts=[]
        if args.rank==0:
            artifacts=[video_file(out/'codec_same_shot_reconstruction.mp4',streamed,24.),
                video_file(out/'codec_hard_cut_reconstruction.mp4',torch.cat((streamed,new_stream),dim=2),24.)]
    result=dict(status='native_fixed_chunk_stream_decoder_pass',rank=args.rank,
        device_name=torch.cuda.get_device_name(device),real_latent_canvas=list(normalized.shape[-2:]),
        source_initial_latents=7,regular_latents=5,initial_pixel_frames=22,regular_pixel_frames=17,
        same_shot_stream_max_abs_error=error,new_shot_stream_max_abs_error=cut_error,
        hard_cut_decoder_state_reset=True,first_shot_frames=streamed.shape[2],new_shot_frames=new_stream.shape[2],
        pending_pixel_frames=5,pending_policy='defer_until_next_same_shot_chunk_or_flush_before_cut',
        old_scene_decoder_prefix_used_at_new_shot=False,ground_truth_reconstruction_only=True,
        artifacts=artifacts,wall_seconds=time.monotonic()-started,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    (out/f'rank{args.rank}_CODEC_PASS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
