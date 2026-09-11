#!/usr/bin/env python3
"""Replay only VAE from saved latents; require full raw-RGB equality before source use."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.history_cache import tensor_sha256
from adapters.longlive_sparse.native_vae_stream import NativeVAEStream
from adapters.longlive_sparse.stream_video_sink import IncrementalVideoSink


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    for name in ('case','assets','source','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    reference=json.loads((args.case/'summary.json').read_text())
    latent=torch.load(args.case/'latents.pt',map_location='cpu',weights_only=True)
    if reference['status']!='pass' or tensor_sha256(latent)!=reference['latent_sha256']:
        raise ValueError('saved native latent/reference integrity required')
    if list(latent.shape)!=[1,128,48,44,80] or latent.dtype!=torch.bfloat16:
        raise ValueError('qualified full509 native latent shape required')
    archives=reference['causal_block_memory']['scene_selection']['archives']
    archive=next(x for x in archives if x['archive_version']==2)
    if archive['source_end']!=48:raise ValueError('qualified source40:48 required')
    sys.path.insert(0,str(args.source));os.chdir(args.assets)
    from utils.wan_5b_wrapper import WanVAEWrapper
    from wan_5b.modules.vae2_2 import unpatchify
    began=time.perf_counter();vae=WanVAEWrapper().to(device='cuda',dtype=torch.bfloat16)
    torch.cuda.synchronize();load_s=time.perf_counter()-began;started=time.perf_counter();pixels=[]
    def observe(frame,rgb):
        if 157<=frame<189:pixels.append(rgb.clone())
    sink=IncrementalVideoSink(args.output/'video.mp4',expected_frames=509,started=started,fps=24,rgb_observer=observe)
    decoder=NativeVAEStream(vae.model,[vae.mean.to(device='cuda',dtype=torch.bfloat16),
        1./vae.std.to(device='cuda',dtype=torch.bfloat16)],unpatchify)
    report=dict(status='running',GPU=torch.cuda.get_device_name(),VAE_load_s=load_s,
        source_case=str(args.case),diffusion_not_rerun=True,scope='offline VAE replay, not a live source observer')
    try:
        for start in range(0,128,8):
            z=latent[:,start:start+8].to('cuda').permute(0,2,1,3,4)
            for raw in decoder.iter_decode(z):
                host=raw.float().clamp_(-1,1).contiguous().cpu()
                sink(host.permute(0,2,1,3,4))
        decoder.finish();result=sink.close();report['pixels']=result
        report['VAE_replay_s']=time.perf_counter()-started
        exact=result['raw_RGB_sha256']==reference['pixels']['raw_RGB_sha256']
        source_pixels=torch.stack(pixels)
        record=dict(archive_version=2,source_start=40,source_end=48,source_phase=archive['source_phase'],
            pixel_start=157,pixel_end=189,source_latent_sha256=tensor_sha256(latent[:,40:48]),
            raw_pixel_bytes_sha256=hashlib.sha256(memoryview(source_pixels.numpy())).hexdigest(),
            pixels=source_pixels,ready_s=time.perf_counter()-started,
            input_origin='offline VAE replay of actual saved native latents')
        with (args.output/'source_raw_rgb.pt').open('xb') as f:
            torch.save(dict(records=[record],complete=exact,pixels_before_lossy_codec=True,
                actual_original_full_raw_RGB_exact=exact,not_a_live_observer=True),f)
        report['source_pixel_witness']=dict(records=[{k:v for k,v in record.items() if k!='pixels'}])
        report['original_raw_RGB_exact']=exact
        if not exact:raise RuntimeError('VAE replay raw RGB differs; source payload is not qualified')
        report['status']='pass'
    except BaseException:
        report.update(status='fail',traceback=traceback.format_exc());raise
    finally:
        (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_pixel_witness','pixels')}))


if __name__=='__main__':main()
