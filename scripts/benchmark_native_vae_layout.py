#!/usr/bin/env python3
"""Bounded decoder-only characterization; approximate weights layout is NOT promoted.

Five warmups and thirty randomized paired measurements, saved native16 prefix.
Timer includes cache reset, eager dispatch, decoder and native float/clamp output;
excludes model load/layout conversion, H2D, CPU hashes, encoding and DiT.
CUDA events measure stream span, not isolated kernel service or HBM counters.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback

import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.native_vae_layout import NativeVAEMemoryFormat
from adapters.longlive_sparse.native_vae_stream import NativeVAEStream
from adapters.longlive_sparse.full_flow_profile import normalize_raw_vae,unit_video_to_rgb
from scripts.gate_native_vae_layout import tensor_bytes,weight_digest


def paired_order(repeats=30,seed=20260910):
    rng=random.Random(seed);rows=[]
    for _ in range(repeats):
        modes=['baseline','weights_only'];rng.shuffle(modes);rows.append(modes)
    return rows


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--gate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    for key in ('source','assets','reference','gate','output'):setattr(args,key,getattr(args,key).resolve())
    if args.output.is_relative_to(args.assets):raise ValueError('read-only assets')
    args.output.mkdir(parents=True,exist_ok=False);report=dict(status='running',measurements=[],warmups=[],first_calls=[])
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        if not torch.cuda.is_available():raise RuntimeError('real CUDA required')
        source_sha=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()
        assert source_sha=='6b36d20ec6f7958d29d11a704dfa64611a9f2572'
        gate=json.loads(args.gate.read_text());ref=json.loads((args.reference/'summary.json').read_text())
        assert gate['status']==ref['status']=='pass'
        assert hashlib.sha256((args.reference/'summary.json').read_bytes()).hexdigest()==gate['reference_summary_sha256']
        assert hashlib.sha256((args.assets/'assets_manifest.json').read_bytes()).hexdigest()==ref['assets_manifest_sha256']
        gate_rows={r['mode']:r for r in gate['rows']}
        assert gate_rows['baseline']['lossless_track_eligible'] and not gate_rows['weights_only']['lossless_track_eligible']
        latent=torch.load(args.reference/'latents.pt',map_location='cpu',weights_only=True)[:,:16].contiguous()
        assert hashlib.sha256(tensor_bytes(latent)).hexdigest()==gate['latent_prefix_sha256']
        sys.path.insert(0,str(args.source));os.chdir(args.assets)
        from utils.wan_5b_wrapper import WanVAEWrapper
        from wan_5b.modules.vae2_2 import unpatchify
        began=time.perf_counter();vae=WanVAEWrapper().to(device='cuda',dtype=torch.bfloat16).eval()
        latent=latent.cuda();torch.cuda.synchronize();loading=time.perf_counter()-began
        scale=[vae.mean.to(device='cuda',dtype=latent.dtype),1./vae.std.to(device='cuda',dtype=latent.dtype)]
        expected_weights=weight_digest(vae.model)

        def decode(*,verify=False):
            digest=hashlib.sha256();count=0
            stream=NativeVAEStream(vae.model,scale,unpatchify)
            for start in range(0,16,8):
                for pixels in stream.iter_decode(latent[:,start:start+8].permute(0,2,1,3,4)):
                    actual=pixels.float().clamp_(-1,1);count+=actual.shape[2]
                    if verify:
                        cpu=actual.permute(0,2,1,3,4).cpu()
                        assert torch.isfinite(cpu).all()
                        digest.update(unit_video_to_rgb(normalize_raw_vae(cpu)).numpy().tobytes())
            stream.finish();assert count==61
            return digest.hexdigest() if verify else None

        def timed(mode):
            with NativeVAEMemoryFormat(vae.model,mode):
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                began=time.perf_counter();start.record();decode();end.record();end.synchronize()
                row=dict(mode=mode,wall_s=time.perf_counter()-began,CUDA_stream_span_ms=start.elapsed_time(end),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            return row

        for mode in ('baseline','weights_only'):
            # First-call algorithm/workspace initialization is kept, never a warm sample.
            report['first_calls'].append(timed(mode))
            with NativeVAEMemoryFormat(vae.model,mode):
                assert weight_digest(vae.model)==expected_weights
                assert decode(verify=True)==gate_rows[mode]['RGB_sha256']
            for i in range(5):report['warmups'].append(dict(repeat=i,**timed(mode)))
        schedule=paired_order()
        for repeat,modes in enumerate(schedule):
            for mode in modes:
                row=dict(repeat=repeat,**timed(mode));report['measurements'].append(row)
                print(json.dumps(row),flush=True)
            (args.output/'progress.json').write_text(json.dumps(dict(completed_pairs=repeat+1,expected_pairs=30))+'\n')
        assert weight_digest(vae.model)==expected_weights
        summaries={}
        for mode in ('baseline','weights_only'):
            values=sorted(r['wall_s'] for r in report['measurements'] if r['mode']==mode)
            summaries[mode]=dict(n=len(values),median_wall_s=statistics.median(values),
                p95_nearest_rank_wall_s=values[28],minimum_wall_s=values[0],maximum_wall_s=values[-1])
        report.update(status='pass',summaries=summaries,
            median_wall_ratio_baseline_over_layout=summaries['baseline']['median_wall_s']/summaries['weights_only']['median_wall_s'],
            order=schedule,order_seed=20260910,warmups_per_mode=5,repeats_per_mode=30,
            numeric_gate_sha256=hashlib.sha256(args.gate.read_bytes()).hexdigest(),
            inherited_error_vs_native_BF16=gate_rows['weights_only']['error_vs_native_BF16_decoder'],
            lossless_candidate=False,candidate_adopted=False,not_full_video_or_end_to_end=True,
            source_sha=source_sha,runner_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
            benchmark_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            GPU=torch.cuda.get_device_name(),torch=torch.__version__,loading_and_H2D_s=loading,
            CUDA_events_are_stream_span_not_kernel_service=True,CPU_hashes_and_layout_setup_excluded=True,
            weight_values_preserved=True)
    except Exception:report.update(status='fail',traceback=traceback.format_exc());raise
    finally:
        (args.output/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
