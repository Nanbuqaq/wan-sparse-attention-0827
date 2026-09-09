#!/usr/bin/env python3
"""Native-resolution decoder layout correctness on a saved 16-latent prefix.

Includes CPU error/hash work; wall values are not speedup measurements.
No new DiT video is generated and no source weight file is modified.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.native_vae_layout import NativeVAEMemoryFormat
from adapters.longlive_sparse.native_vae_stream import NativeVAEStream
from adapters.longlive_sparse.full_flow_profile import normalize_raw_vae,unit_video_to_rgb


def tensor_bytes(x):return x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def weight_digest(model):
    digest=hashlib.sha256()
    for name,parameter in model.named_parameters():
        digest.update(name.encode());digest.update(str(parameter.dtype).encode())
        digest.update(json.dumps(list(parameter.shape)).encode());digest.update(tensor_bytes(parameter))
    return digest.hexdigest()


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    for key in ('source','assets','reference','output'):setattr(args,key,getattr(args,key).resolve())
    if args.output.is_relative_to(args.assets):raise ValueError('output cannot be placed in the read-only assets tree')
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',scope='saved_latent_VAE_layout_numeric_gate_not_latency_benchmark',rows=[])
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        if not torch.cuda.is_available():raise RuntimeError('real CUDA required')
        source_sha=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()
        if source_sha!='6b36d20ec6f7958d29d11a704dfa64611a9f2572':raise ValueError('native source lock changed')
        ref=json.loads((args.reference/'summary.json').read_text());assert ref['status']=='pass'
        manifest=args.assets/'assets_manifest.json'
        assert hashlib.sha256(manifest.read_bytes()).hexdigest()==ref['assets_manifest_sha256']
        latent=torch.load(args.reference/'latents.pt',map_location='cpu',weights_only=True)
        assert list(latent.shape)==[1,128,48,44,80]
        latent=latent[:,:16].contiguous()
        report.update(source_sha=source_sha,runner_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      reference_summary_sha256=hashlib.sha256((args.reference/'summary.json').read_bytes()).hexdigest(),
                      latent_prefix_sha256=hashlib.sha256(tensor_bytes(latent)).hexdigest(),
                      GPU=torch.cuda.get_device_name(),torch=torch.__version__,latent_shape=list(latent.shape),expected_pixels=61)
        sys.path.insert(0,str(args.source));os.chdir(args.assets)
        from utils.wan_5b_wrapper import WanVAEWrapper
        from wan_5b.modules.vae2_2 import unpatchify
        started=time.perf_counter();vae=WanVAEWrapper().to(device='cuda',dtype=torch.bfloat16).eval()
        latent=latent.cuda();torch.cuda.synchronize();report['loading_and_input_H2D_s']=time.perf_counter()-started
        original_weights=weight_digest(vae.model)
        reference=vae.decode_to_pixel(latent).float().cpu();assert reference.shape==(1,61,3,704,1280)
        assert torch.isfinite(reference).all()
        ref_rgb=unit_video_to_rgb(normalize_raw_vae(reference));expected_rgb=hashlib.sha256(ref_rgb.numpy().tobytes()).hexdigest()
        del ref_rgb
        scale=[vae.mean.to(device='cuda',dtype=latent.dtype),1./vae.std.to(device='cuda',dtype=latent.dtype)]
        for mode in NativeVAEMemoryFormat.MODES:
            offset=0;maximum=error2=ref2=actual2=dot=0.;exact=True;rgb=hashlib.sha256()
            context=NativeVAEMemoryFormat(vae.model,mode);setup=time.perf_counter()
            with context:
                torch.cuda.synchronize();setup_s=time.perf_counter()-setup
                assert weight_digest(vae.model)==original_weights
                stream=NativeVAEStream(vae.model,scale,unpatchify)
                torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
                for start in range(0,16,8):
                    for decoded in stream.iter_decode(latent[:,start:start+8].permute(0,2,1,3,4)):
                        actual=decoded.float().clamp_(-1,1).permute(0,2,1,3,4).cpu()
                        if not torch.isfinite(actual).all():raise ValueError('nonfinite layout output')
                        count=actual.shape[1];base=reference[:,offset:offset+count]
                        exact=exact and torch.equal(actual,base)
                        a,r=actual.double(),base.double();diff=a-r
                        maximum=max(maximum,float(diff.abs().max()));error2+=float(diff.square().sum())
                        ref2+=float(r.square().sum());actual2+=float(a.square().sum());dot+=float((a*r).sum())
                        rgb.update(unit_video_to_rgb(normalize_raw_vae(actual)).numpy().tobytes());offset+=count
                stream.finish();torch.cuda.synchronize();elapsed=time.perf_counter()-started
                peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
            assert offset==61 and weight_digest(vae.model)==original_weights
            error=dict(max_abs=maximum,relative_l2=math.sqrt(error2/max(ref2,1e-24)),
                       one_minus_cosine=max(0.,1-min(1.,dot/max(math.sqrt(ref2*actual2),1e-24))))
            row=dict(**context.record(),frames=offset,layout_setup_s=setup_s,diagnostic_wall_s_including_CPU_errors=elapsed,
                     GPU_peak_allocated_bytes=peak,GPU_peak_reserved_bytes=reserved,weight_values_unchanged=True,
                     raw_float_exact=exact,RGB_exact=rgb.hexdigest()==expected_rgb,RGB_sha256=rgb.hexdigest(),
                     error_vs_native_BF16_decoder=error,lossless_track_eligible=exact and rgb.hexdigest()==expected_rgb)
            report['rows'].append(row);print(json.dumps(row),flush=True)
            if mode=='baseline' and not row['lossless_track_eligible']:
                raise ValueError('baseline stream changed native prefix; do not interpret layout comparisons')
        report.update(status='pass',unchanged_parameter_values=True,reference_RGB_sha256=expected_rgb,
                      not_FP32_model_reference=True,not_full_video_qualification=True,
                      candidate_adopted=False,approximate_variants_not_promoted=True)
    except Exception:
        report.update(status='fail',traceback=traceback.format_exc());raise
    finally:
        (args.output/'gate.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
