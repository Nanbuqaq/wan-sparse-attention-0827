#!/usr/bin/env python3
"""Bounded effective-bandwidth and BF16 GEMM calibration; no video or algorithm change."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import torch


def bind_process_threads(cpus):
    for tid in Path('/proc/self/task').iterdir():
        try:os.sched_setaffinity(int(tid.name),cpus)
        except ProcessLookupError:pass


def numa_map(tensor):
    pointer=tensor.data_ptr();mapping=None
    for line in Path('/proc/self/maps').read_text().splitlines():
        start,end=(int(v,16) for v in line.split()[0].split('-'))
        if start<=pointer<end:mapping=start;break
    if mapping is None:return dict(status='unavailable')
    for line in Path('/proc/self/numa_maps').read_text().splitlines():
        if int(line.split()[0],16)==mapping:
            return dict(status='observed_VMA_not_tensor_exclusive',VMA_description=line,
                        per_tensor_page_placement_not_guaranteed=True)
    return dict(status='unavailable')


def measure(action,*,gpu=True,repeats=30):
    for _ in range(5):action()
    if gpu:torch.cuda.synchronize()
    samples=[]
    for _ in range(repeats):
        if gpu:torch.cuda.synchronize()
        start=time.perf_counter();action()
        if gpu:torch.cuda.synchronize()
        samples.append(time.perf_counter()-start)
    return dict(median_s=statistics.median(samples),p95_s=float(np.percentile(samples,95)),samples_s=samples,
                warmup=5,repeats=repeats,scope='synchronized_host_wall_including_CPU_pack_if_present')


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--local-cpus',default='16,17');p.add_argument('--remote-cpus',default='112,113')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available():raise RuntimeError('real GPU required')
    local={int(x) for x in args.local_cpus.split(',')};remote={int(x) for x in args.remote_cpus.split(',')}
    allowed=os.sched_getaffinity(0)
    if not (local|remote)<=allowed:raise ValueError('requested CPU outside caller affinity')
    bind_process_threads(local);torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20260908)
    prop=torch.cuda.get_device_properties(0)
    info={key:getattr(prop,key) for key in dir(prop) if not key.startswith('_') and isinstance(getattr(prop,key),(int,str,bool))}
    result=dict(status='running',created_UTC=datetime.now(timezone.utc).isoformat(),gpu_properties=info,
        torch=str(torch.__version__),physical_gpu=os.environ.get('WAN_SPARSE_PHYSICAL_GPU'),
        initial_allowed_CPUs=sorted(allowed),local_CPUs=sorted(local),remote_CPUs=sorted(remote),
        torch_CPU_threads=2,explicit_NUMA_membind_not_used=True,throughput_is_workload_specific_not_architectural_peak=True,
        lscpu=subprocess.check_output(['lscpu'],text=True),topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),
        bandwidth=[],gemms=[])
    def save(): (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    for location,cpus in (('local_CPU_affinity',local),('remote_CPU_affinity',remote)):
        bind_process_threads(cpus)
        for mib in (32,256):
            n=mib*1024**2;host=torch.empty(n,dtype=torch.uint8).random_(0,256)
            pin=torch.empty(n,dtype=torch.uint8,pin_memory=True);pin.copy_(host)
            device=torch.empty(n,dtype=torch.uint8,device='cuda');device.copy_(host)
            back=torch.empty_like(host)
            actions={
                'pageable_H2D':lambda:device.copy_(host,non_blocking=True),
                'pinned_H2D':lambda:device.copy_(pin,non_blocking=True),
                'CPU_pack_plus_pinned_H2D':lambda:(pin.copy_(host),device.copy_(pin,non_blocking=True)),
                'pinned_D2H':lambda:pin.copy_(device,non_blocking=True),
                'pageable_D2H':lambda:back.copy_(device,non_blocking=False)}
            for name,action in actions.items():
                row=measure(action);row.update(operation=name,affinity=location,payload_bytes=n,
                    effective_payload_GBps=n/row['median_s']/1e9,
                    correctness=bool(torch.equal(device.cpu(),host) and (torch.equal(back,host) if name=='pageable_D2H' else torch.equal(pin,host))),
                    host_VMA=numa_map(host),pinned_VMA=numa_map(pin))
                if not row['correctness']:raise RuntimeError('copy correctness failed')
                result['bandwidth'].append(row);save();print(f'{location} {mib}MiB {name}: {row["effective_payload_GBps"]:.3f} GB/s',flush=True)
            del host,pin,device,back
        # Larger than the host's aggregate512MiB L3; input+output1GiB each.
        host=torch.empty(1024**3,dtype=torch.uint8).fill_(73);back=torch.empty_like(host)
        row=measure(lambda:back.copy_(host),gpu=False);row.update(operation='CPU_contiguous_copy',affinity=location,
            payload_bytes=host.numel(),effective_payload_GBps=host.numel()/row['median_s']/1e9,
            modeled_read_plus_write_GBps=2*host.numel()/row['median_s']/1e9,
            correctness=bool(torch.equal(host,back)),host_VMA=numa_map(host))
        result['bandwidth'].append(row);del host,back;save()
    bind_process_threads(local)
    source=torch.empty(1024**3,dtype=torch.uint8,device='cuda').fill_(57);target=torch.empty_like(source)
    row=measure(lambda:target.copy_(source));row.update(operation='GPU_D2D_copy',payload_bytes=source.numel(),
        effective_payload_GBps=source.numel()/row['median_s']/1e9,
        modeled_read_plus_write_GBps=2*source.numel()/row['median_s']/1e9,
        physical_DRAM_transactions_not_measured=True,correctness=bool(torch.equal(source,target)))
    result['bandwidth'].append(row);del source,target;save()
    shapes=[(4680,1536,1536),(4680,1536,8960),(4680,8960,1536),(4680,1536,256),
            (4680,256,1536),(4680,8960,256),(4680,256,8960),(512,4096,1536),(8192,8192,8192)]
    for m,k,n in shapes:
        a=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)/math.sqrt(k)
        b=torch.randn(k,n,device='cuda',dtype=torch.bfloat16);out=torch.empty(m,n,device='cuda',dtype=torch.bfloat16)
        row=measure(lambda:torch.mm(a,b,out=out));ref=a[:16].float()@b[:,:16].float();actual=out[:16,:16].float()
        row.update(M=m,K=k,N=n,dtype='BF16',FLOPs=2*m*k*n,effective_TFLOPs=2*m*k*n/row['median_s']/1e12,
            sample_max_abs=float((actual-ref).abs().max()),sample_relative_l2=float(torch.linalg.vector_norm(actual-ref)/torch.linalg.vector_norm(ref)),
            torch_mm_preallocated_output=True,not_full_module_or_end_to_end_speed=True)
        if row['sample_max_abs']>.02 or row['sample_relative_l2']>.01:raise RuntimeError('GEMM numerical gate failed')
        result['gemms'].append(row);save();print(f'GEMM {m},{k},{n}: {row["effective_TFLOPs"]:.3f} TFLOPs/s',flush=True)
        del a,b,out,ref,actual
    result['telemetry_after_work']=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,pcie.link.gen.current,pcie.link.width.current,clocks.sm,clocks.mem,power.draw','--format=csv,noheader'],text=True)
    result['status']='pass';save()


if __name__=='__main__':main()
