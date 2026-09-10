#!/usr/bin/env python3
"""Exact-moment real-QKV gate plus synchronized paired preparation timings."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.native_resident_history import summarize_frame
from adapters.longlive_sparse.native_summary_vectorized import summarize_frame_vectorized


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('real GPU required')
    data=torch.load(args.capture,weights_only=True,map_location='cpu',mmap=True);checks=[];bench=[]
    for index,record in enumerate(data['records']):
        k,v=[record[name][0].cuda() for name in ('k','v')]
        if k.shape!=(28160,24,128):raise ValueError('native880/head24/d128 capture required')
        for frame in range(32):
            a,b=k[frame*880:(frame+1)*880],v[frame*880:(frame+1)*880]
            scalar=summarize_frame(a,b);vector=summarize_frame_vectorized(a,b)
            checks.append(dict(capture_row=index,frame=frame,all_moments_and_counts_bitwise_equal=all(torch.equal(x,y) for x,y in zip(scalar,vector)),
                K_max_abs=float((scalar[0]-vector[0]).abs().max()),V_max_abs=float((scalar[1]-vector[1]).abs().max())))
        if index==0:
            # Same32 already-resident frames; no file load/teacher/verification inside timing.
            def prepare(fn):return [fn(k[f*880:(f+1)*880],v[f*880:(f+1)*880]) for f in range(32)]
            funcs={'scalar':summarize_frame,'vectorized':summarize_frame_vectorized}
            for _ in range(3):
                for fn in funcs.values():prepared=prepare(fn)
            rng=random.Random(20260910)
            for repeat in range(20):
                names=list(funcs);rng.shuffle(names)
                for name in names:
                    torch.cuda.synchronize();began=time.perf_counter()
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();prepared=prepare(funcs[name]);end.record()
                    launch_finished=time.perf_counter();end.synchronize()
                    bench.append(dict(repeat=repeat,backend=name,synchronized_wall_ms=(time.perf_counter()-began)*1000,
                        host_launch_scope_ms=(launch_finished-began)*1000,CUDA_stream_interval_ms=start.elapsed_time(end)))
            del prepared
        del k,v,scalar,vector
    exact=all(r['all_moments_and_counts_bitwise_equal'] for r in checks)
    medians={name:statistics.median(r['synchronized_wall_ms'] for r in bench if r['backend']==name) for name in ('scalar','vectorized')}
    with args.capture.open('rb') as h:sha=hashlib.file_digest(h,'sha256').hexdigest()
    out=dict(status='exact_gate_pass' if exact else 'numerical_mismatch_no_runtime_integration',checks=checks,timings=bench,
        median_synchronized_wall_ms=medians,capture_sha256=sha,GPU=torch.cuda.get_device_name(0),
        prototype_or_route_semantics_changed=False,full_video_equivalence_not_yet_checked=True,
        CUDA_stream_interval_includes_launch_gaps_not_pure_kernel_service=True,
        host_launch_scope_is_not_pure_CPU_arithmetic=True,benchmark_scope='32 resident native frames, summary creation only')
    (args.output/'summary_vectorization.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'status':out['status'],'frame_checks':len(checks),'exact':sum(r['all_moments_and_counts_bitwise_equal'] for r in checks),'medians':medians}))
    if not exact:raise SystemExit(1)


if __name__=='__main__':main()
