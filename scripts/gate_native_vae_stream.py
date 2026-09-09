#!/usr/bin/env python3
"""Real-GPU native decoder equivalence on saved latents; no DiT generation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.native_vae_stream import NativeVAEStream
from adapters.longlive_sparse.full_flow_profile import normalize_raw_vae,unit_video_to_rgb


def rgb_digest(shape):
    result=hashlib.sha256();result.update(str(torch.uint8).encode());result.update(json.dumps(list(shape)).encode());return result


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('real GPU required')
    source_sha=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()
    assert source_sha=='6b36d20ec6f7958d29d11a704dfa64611a9f2572'
    summary=json.loads((args.reference/'summary.json').read_text());assert summary['status']=='pass'
    manifest=args.assets/'assets_manifest.json'
    assert json.loads(manifest.read_text())['status']=='pass'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==summary['assets_manifest_sha256']
    latent=torch.load(args.reference/'latents.pt',map_location='cpu',weights_only=True)
    assert list(latent.shape)==[1,64,48,16,32]
    sys.path.insert(0,str(args.source));os.chdir(args.assets)
    from utils.wan_5b_wrapper import WanVAEWrapper
    from wan_5b.modules.vae2_2 import unpatchify
    started=time.perf_counter();vae=WanVAEWrapper().to(device='cuda',dtype=torch.bfloat16).eval()
    latent=latent.cuda();torch.cuda.synchronize();loading_s=time.perf_counter()-started
    started=time.perf_counter();reference=vae.decode_to_pixel(latent).cpu();reference_s=time.perf_counter()-started
    reference_rgb=unit_video_to_rgb(normalize_raw_vae(reference))
    expected_shape=tuple(reference_rgb.shape);digest=rgb_digest(expected_shape);digest.update(reference_rgb.numpy().tobytes())
    batch_digest=digest.hexdigest();original_rgb_exact=batch_digest==summary['pixels']['raw_RGB_sha256']
    scale=[vae.mean.to(device='cuda',dtype=latent.dtype),1./vae.std.to(device='cuda',dtype=latent.dtype)]
    rows=[]
    for chunk_size in (8,7,13):
        stream=NativeVAEStream(vae.model,scale,unpatchify);offset=0;max_abs=0.;error2=ref2=actual2=dot=0.;exact=True
        digest=rgb_digest(expected_shape);first_cpu_pixels_s=None;started=time.perf_counter()
        for start in range(0,latent.shape[1],chunk_size):
            chunk=latent[:,start:start+chunk_size].permute(0,2,1,3,4)
            for decoded in stream.iter_decode(chunk):
                actual=decoded.float().clamp_(-1,1).permute(0,2,1,3,4).cpu()
                if first_cpu_pixels_s is None:first_cpu_pixels_s=time.perf_counter()-started
                count=actual.shape[1];base=reference[:,offset:offset+count]
                exact=exact and torch.equal(actual,base)
                a=actual.double();r=base.double();diff=a-r
                max_abs=max(max_abs,float(diff.abs().max()));error2+=float(diff.square().sum());ref2+=float(r.square().sum())
                actual2+=float(a.square().sum());dot+=float((a*r).sum())
                digest.update(unit_video_to_rgb(normalize_raw_vae(actual)).numpy().tobytes());offset+=count
        stream.finish();elapsed=time.perf_counter()-started
        assert offset==253
        error=dict(max_abs=max_abs,relative_l2=math.sqrt(error2/max(ref2,1e-24)),
            one_minus_cosine=max(0.,1-min(1.,dot/max(math.sqrt(ref2*actual2),1e-24))))
        passed=error['max_abs']<=.02 and error['relative_l2']<=.01 and error['one_minus_cosine']<=.001
        rows.append(dict(chunk_size=chunk_size,latent_frames=64,pixel_frames=offset,numeric_gate=passed,
            raw_float_bitwise_exact=exact,RGB_sha256=digest.hexdigest(),RGB_matches_recorded=digest.hexdigest()==summary['pixels']['raw_RGB_sha256'],
            error_vs_native_BF16_decode_float_output=error,first_CPU_pixel_group_s=first_cpu_pixels_s,
            wall_including_CPU_error_and_hash_s=elapsed))
        print(json.dumps(rows[-1]),flush=True)
    result=dict(status='pass' if original_rgb_exact and all(r['numeric_gate'] for r in rows) else 'fail',
        GPU=torch.cuda.get_device_name(),torch=torch.__version__,source_sha=source_sha,
        source_files_sha256={name:hashlib.sha256((args.source/name).read_bytes()).hexdigest() for name in ('utils/wan_5b_wrapper.py','wan_5b/modules/vae2_2.py')},
        reference_summary_sha256=hashlib.sha256((args.reference/'summary.json').read_bytes()).hexdigest(),
        batch_replay_RGB_matches_recorded=original_rgb_exact,batch_replay_RGB_sha256=batch_digest,
        loading_s=loading_s,batch_decode_and_D2H_s=reference_s,rows=rows,
        original_native_WanVAE_has_cached_decode=hasattr(vae.model,'cached_decode'),
        unchanged_model_weights=True,not_a_two_GPU_overlap_or_video_speedup_test=True,
        reference_is_native_BF16_decode_not_FP32_model=True)
    (args.output/'gate.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'])))
    if result['status']!='pass':raise RuntimeError('native streaming VAE numerical/recorded RGB gate failed; keep artifacts')


if __name__=='__main__':main()
