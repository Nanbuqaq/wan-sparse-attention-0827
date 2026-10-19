"""VAE-only corrected deliveries in bounded batches; original DiT is reused."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import wave

import torch
from h3_buffered_video_decode import BufferedVideoDecoder
from h3_codec_regression import sha,atomic
from h3_frame_writer import FrameWriter


def selected_video(saved):
    ids=saved['generated_ids']
    if ids!=list(range(len(ids))):raise ValueError('Source generation has missing latent positions')
    return saved['video'][:,:len(ids)]


def render(task,vae,stage,out):
    if sha(task['source_row'])!=task['source_row_sha256']:raise ValueError('Frozen source provenance changed')
    if sha(task['source_latents'])!=task['source_latents_sha256']:raise ValueError('Original generated archive changed')
    saved=torch.load(task['source_latents'],map_location='cpu',weights_only=True,mmap=True)
    video=selected_video(saved)
    if video.shape[1]!=task['video_latents'] or not torch.isfinite(video).all():raise ValueError('Incomplete or invalid original video latents')
    folder=out/task['task_id'];folder.mkdir(exist_ok=True);result_path=folder/'RESULT.json'
    if result_path.exists():
        old=json.loads(result_path.read_text())
        if old['task']!=task or sha(old['artifact']['path'])!=old['artifact']['sha256']:raise ValueError('Existing corrected delivery changed')
        return
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic();decoder=BufferedVideoDecoder(vae,'cuda');path=folder/'video.mp4'
    with torch.inference_mode(),FrameWriter(path,task['fps'],ffmpeg=str(stage/'media_tools/bin/ffmpeg'),encoder_threads=2) as writer:
        at=0
        for count in task['generation_parts']:
            value=decoder.push(video[None,:,at:at+count]);at+=count
            if value is not None:writer.append(value)
        if at!=video.shape[1]:raise ValueError('Original generation parts do not consume all latents')
        tail=decoder.finish()
        if tail is not None:writer.append(tail)
        artifact=writer.artifact()
    if artifact['frames']!=task['frames']:raise ValueError('Corrected decoder changed the output duration')
    video_seconds=time.monotonic()-started;audio_start=time.monotonic();new_audio_decode=False
    ffmpeg=str(stage/'media_tools/bin/ffmpeg');mux=folder/'mux.mp4'
    if task.get('original_video'):
        if sha(task['original_video'])!=task['original_video_sha256']:raise ValueError('Original movie changed')
        subprocess.run([ffmpeg,'-v','error','-nostdin','-y','-i',str(path),'-i',task['original_video'],
                        '-map','0:v:0','-map','1:a:0','-c','copy',str(mux)],check=True,timeout=120)
    else:
        # Proven complete but undelivered model output: offline audio decode,
        # never rerun the DiT and never fill missing final-chunk timings.
        from diffusers import AutoencoderKLMiniMaxH3Audio
        new_audio_decode=True;audio_vae=AutoencoderKLMiniMaxH3Audio.from_pretrained(str(stage/'h3-base'),subfolder='audio_vae').to('cuda').eval()
        mean=torch.tensor(audio_vae.config.latents_mean,device='cuda').view(1,-1,1)
        std=torch.tensor(audio_vae.config.latents_std,device='cuda').view(1,-1,1)
        with torch.inference_mode():audio=audio_vae.decode(saved['audio'].to('cuda')*std+mean,return_dict=False)[0].float()
        if not torch.isfinite(audio).all():raise ValueError('Recovered audio is nonfinite')
        pcm=(audio.permute(1,0,2)[0].clamp(-1,1).cpu().T*32767).round().short().contiguous().numpy().tobytes()
        wav=folder/'audio.wav'
        with wave.open(str(wav),'wb') as stream:
            stream.setnchannels(2);stream.setsampwidth(2);stream.setframerate(32000);stream.writeframes(pcm)
        subprocess.run([ffmpeg,'-v','error','-nostdin','-y','-i',str(path),'-i',str(wav),'-c:v','copy','-c:a','aac',
                        '-af','apad','-t',str(task['frames']/task['fps']),str(mux)],check=True,timeout=120)
        del audio_vae,audio;torch.cuda.empty_cache()
    mux.replace(path);artifact.update(sha256=sha(path),audio_muxed=True)
    result=dict(schema='h3_corrected_codec_delivery_v1',status='complete',task=task,artifact=artifact,
        source_latents_sha256=sha(task['source_latents']),source_row_sha256=task['source_row_sha256'],
        video_decode_contract='continuous',video_decoder_version='buffered_native_equivalent_v1',
        codec_summary=decoder.last_summary,DiT_forward_calls=0,new_audio_decode=new_audio_decode,
        video_decode_export_seconds=video_seconds,audio_mux_seconds=time.monotonic()-audio_start,
        delivery_elapsed_seconds=time.monotonic()-started,encoder_threads=2,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        hardware=torch.cuda.get_device_name(),offline_audio_decode_only=True,formal_memory_quality_claim=False)
    resolved=folder/'RESOLVED_CODEC_CONFIG.json'
    atomic(resolved,dict(task=task,codec='continuous',decoder='buffered_native_equivalent_v1',encoder_threads=2,
                        VAE_files_sha256=json.loads((stage/'PREP_READY.json').read_text())['VAE_files_sha256'],DiT_forward_calls=0))
    result.update(resolved_config_path=str(resolved),resolved_config_sha256=sha(resolved))
    atomic(result_path,result)


def main():
    p=argparse.ArgumentParser();p.add_argument('--lane',type=int,required=True);a=p.parse_args()
    from train_h3_v2_local_history_ddp import bootstrap
    bootstrap()
    from diffusers import AutoencoderKLMiniMaxH3
    stage=Path(os.environ['INFER_WEIGHTS_DIR']);out=Path(os.environ['INFER_OUTPUT_DIR'])
    plan=json.loads((stage/'TASKS.json').read_text());lanes=plan['lanes']
    receipt=json.loads((stage/'PREP_READY.json').read_text())
    if {name:sha(stage/'h3-base/vae'/name) for name in receipt['VAE_files_sha256']}!=receipt['VAE_files_sha256']:
        raise ValueError('VAE files changed after CPU preparation')
    assigned=[t for t in plan['tasks'] if t['lane']==a.lane]
    if not 4<=lanes<=8 or not assigned:raise ValueError('Use useful4..8GPU lanes')
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    vae=AutoencoderKLMiniMaxH3.from_pretrained(str(stage/'h3-base'),subfolder='vae').to('cuda').eval()
    reports=[]
    for task in assigned:
        try:
            render(task,vae,stage,out);reports.append(dict(task_id=task['task_id'],status='complete'))
            print('CORRECTED_CODEC_DELIVERY',task['task_id'],flush=True)
        except Exception as error:
            folder=out/task['task_id'];folder.mkdir(exist_ok=True)
            failure=dict(task_id=task['task_id'],error=str(error),traceback=traceback.format_exc())
            atomic(folder/'DELIVERY_ERROR.json',failure);reports.append(dict(task_id=task['task_id'],status='error'))
    atomic(out/f'LANE{a.lane}.json',dict(results=reports,DiT_forward_calls=0))
    if any(r['status']!='complete' for r in reports):raise RuntimeError('Some codec deliveries failed; keep completed outputs and inspect errors')


if __name__=='__main__':main()
