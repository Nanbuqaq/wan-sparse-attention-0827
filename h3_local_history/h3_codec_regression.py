"""InferHub VAE-only regression on immutable latents; never loads a DiT."""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import av
import numpy as np
import torch
from h3_buffered_video_decode import BufferedVideoDecoder


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def pixels(decoded):
    mean=torch.tensor([.485,.456,.406],device=decoded.device).view(1,3,1,1,1)
    std=torch.tensor([.229,.224,.225],device=decoded.device).view(1,3,1,1,1)
    return ((decoded.float()*std+mean).clamp(0,1)*255).round().byte()[0].permute(1,2,3,0).cpu().numpy()


def emit(path,rgb,task):
    silent=path.with_name(path.stem+'_silent.mp4')
    clock=Fraction(*task['time_base']);rate=Fraction(task['fps'])
    if len(rgb)!=len(task['frame_pts']):raise ValueError('Decoder changed the requested frame count')
    with av.open(str(silent),'w') as container:
        stream=container.add_stream('libx264',rate=rate)
        stream.width=rgb.shape[2];stream.height=rgb.shape[1];stream.pix_fmt='yuv420p';stream.time_base=clock
        stream.codec_context.time_base=clock;stream.options=dict(crf='18',preset='fast',threads='2')
        for frame_id,x in enumerate(rgb):
            frame=av.VideoFrame.from_ndarray(x,format='rgb24');frame.pts=task['frame_pts'][frame_id];frame.time_base=clock
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    if task.get('original_video') and task.get('audio_present'):
        subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-i',str(silent),'-i',task['original_video'],
                        '-map','0:v:0','-map','1:a:0','-c','copy',str(path)],check=True,timeout=120)
        silent.unlink()
    else:silent.replace(path)
    return dict(path=str(path),sha256=sha(path),frames=len(rgb),audio_copied=bool(task.get('audio_present')),
                frame_rgb_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in rgb],
                frame_channel_means=[x.mean((0,1)).tolist() for x in rgb])


def decode_whole(decoder,segments,mode):
    if mode=='continuous':return decoder.vae.decode(torch.cat(segments,2)*decoder.std+decoder.mean,return_dict=False)[0]
    return torch.cat([decoder.vae.decode(z*decoder.std+decoder.mean,return_dict=False)[0] for z in segments],2)


def decode_buffered(decoder,segments,mode,parts,fixed5=False):
    groups=[torch.cat(segments,2)] if mode=='continuous' else segments
    plans=[sum(parts,[])] if mode=='continuous' else parts
    out=[];summaries=[]
    for z,plan in zip(groups,plans):
        if fixed5:plan=[5]*(z.shape[2]//5)+([z.shape[2]%5] if z.shape[2]%5 else [])
        at=0
        for n in plan:
            value=decoder.push(z[:,:,at:at+n]);at+=n
            if value is not None:out.append(value)
        if at!=z.shape[2]:raise ValueError('Incomplete generation partition')
        tail=decoder.finish()
        if tail is not None:out.append(tail)
        summaries.append(decoder.last_summary)
    return torch.cat(out,2),summaries


def main():
    p=argparse.ArgumentParser();p.add_argument('--lane',type=int,required=True);a=p.parse_args()
    from train_h3_v2_local_history_ddp import bootstrap
    bootstrap()
    from diffusers import AutoencoderKLMiniMaxH3
    stage=Path(os.environ['INFER_WEIGHTS_DIR']);out=Path(os.environ['INFER_OUTPUT_DIR'])
    manifest=json.loads((stage/'TASKS.json').read_text());task=manifest['tasks'][a.lane]
    prep=json.loads((stage/'PREP_READY.json').read_text())
    actual_vae_hashes={name:sha(stage/'h3-base/vae'/name) for name in prep['vae_files']}
    if actual_vae_hashes!=prep['vae_files']:raise ValueError('VAE files changed after CPU preparation')
    torch.set_num_threads(2);torch.set_grad_enabled(False);device=torch.device('cuda')
    if sha(task['latents'])!=task['latents_sha256']:raise ValueError('Frozen decoder input changed')
    saved=torch.load(task['latents'],map_location='cpu',weights_only=True)
    segments=[z.to(device) for z in saved.get('segments',[])]
    if not all(torch.isfinite(z).all() for z in segments):raise ValueError('Nonfinite input latent')
    folder=out/task['case_id'];folder.mkdir(exist_ok=True)
    started=time.monotonic();vae=AutoencoderKLMiniMaxH3.from_pretrained(str(stage/'h3-base'),subfolder='vae').to(device).eval()
    decoder=BufferedVideoDecoder(vae,device);variants=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        if task.get('clean_codec_control'):
            source_tensor=saved['source_rgb'].to(device).float().permute(3,0,1,2)[None]/255
            rgb_mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1,1)
            rgb_std=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1,1)
            z=(vae.encode((source_tensor-rgb_mean)/rgb_std).latent_dist.mode()-decoder.mean)/decoder.std
            if z.shape[2]!=32:raise ValueError('107-frame clean codec control must encode to32 latents')
            segments=[z[:,:,:12],z[:,:,12:22],z[:,:,22:]]
            torch.save(dict(segments=[v.cpu() for v in segments]),folder/'ENCODED_CONTROL.pt')
        reference=decode_whole(decoder,segments,task['codec_mode'])
        legacy=[]
        for i,z in enumerate(segments):
            if task['legacy_reset_fake_prefix'] and i:
                prefix=z[:,:,:1].repeat(1,1,2,1,1)
                legacy.append(vae.decode(torch.cat([prefix,z],2)*decoder.std+decoder.mean,return_dict=False)[0][:,:,5:])
            else:legacy.append(vae.decode(z*decoder.std+decoder.mean,return_dict=False)[0])
        old=torch.cat(legacy,2)
        for name,value,details in [('legacy',old,None),('whole_contract',reference,None)]:
            rgb=pixels(value);artifact=emit(folder/(name+'.mp4'),rgb,task)
            variants.append(dict(policy=name,artifact=artifact));del rgb
        del old
        gates=[]
        for name,fixed in [('buffered_contract',False),('fixed5_buffered_contract',True)]:
            value,summaries=decode_buffered(decoder,segments,task['codec_mode'],saved['generation_parts'],fixed)
            if value.shape!=reference.shape:raise ValueError('Buffered decode changed full output shape')
            difference=(value.float()-reference.float()).abs();maximum=float(difference.max());mean=float(difference.mean());del difference
            rgb=pixels(value);ref_rgb=pixels(reference)
            pixel_max=int(np.abs(rgb.astype(np.int16)-ref_rgb.astype(np.int16)).max())
            artifact=emit(folder/(name+'.mp4'),rgb,task);del rgb,ref_rgb,value
            passed=maximum<=1e-4 and pixel_max<=1
            gate=dict(policy=name,normalized_maxabs=maximum,normalized_mean_abs=mean,RGB_maxabs=pixel_max,
                      predeclared_normalized_atol=1e-4,predeclared_RGB_atol=1,passed=passed)
            gates.append(gate);variants.append(dict(policy=name,artifact=artifact,whole_decode_gate=gate,buffer_summaries=summaries))
        if task.get('clean_codec_control'):
            independent=[];at=0;valid=saved['source_shot_frames']
            for count in valid:
                x=source_tensor[:,:,at:at+count];at+=count
                if count<39:x=torch.cat([x,x[:,:,-1:].repeat(1,1,39-count,1,1)],2)
                encoded=(vae.encode((x-rgb_mean)/rgb_std).latent_dist.mode()-decoder.mean)/decoder.std
                if encoded.shape[2]!=12:raise ValueError('Declared39-frame independent shot must encode to12 latents')
                independent.append(encoded)
            expected=decode_whole(decoder,independent,'independent_shots')
            actual,summaries=decode_buffered(decoder,independent,'independent_shots',[[12]]*3,True)
            maximum=float((actual.float()-expected.float()).abs().max());passed=maximum<=1e-4
            gates.append(dict(policy='new_data_independent_shots',normalized_maxabs=maximum,passed=passed,predeclared_normalized_atol=1e-4))
            for name,value in [('new_data_independent_whole',expected),('new_data_independent_fixed5',actual)]:
                value=torch.cat([x[:,:,:count] for x,count in zip(value.split(39,dim=2),valid)],2)
                artifact=emit(folder/(name+'.mp4'),pixels(value),task)
                variants.append(dict(policy=name,artifact=artifact,valid_source_frames=valid,
                    per_shot_encoded_frames=[39,39,39],input_tail_padding_frames=[39-count for count in valid],
                    removal_is_declared_codec_input_padding_not_anomaly_filter=True,buffer_summaries=summaries))
            torch.save(dict(segments=[v.cpu() for v in independent],valid_frames=valid),folder/'INDEPENDENT_ENCODED_CONTROL.pt')
        result=dict(status='passed' if all(g['passed'] for g in gates) else 'numerical_difference_requires_diagnosis',
            case_id=task['case_id'],task=task,variants=variants,gates=gates,DiT_evaluations=0,
            saved_latents_unchanged_sha256=sha(task['latents']),hardware=torch.cuda.get_device_name(),
            elapsed_seconds=time.monotonic()-started,peak_HBM_gib=torch.cuda.max_memory_allocated()/2**30,
            VAE_files_sha256=actual_vae_hashes,
            quality_pass_not_inferred_from_numeric_gate=True,official_InferHub_only=True)
        atomic(folder/'RESULT.json',result)
    if not all(g['passed'] for g in gates):raise RuntimeError('Whole/buffer equivalence gate did not pass')


if __name__=='__main__':main()
