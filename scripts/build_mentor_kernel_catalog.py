#!/usr/bin/env python3
"""Nsight launch resources, not measured occupancy or hardware transactions."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3


def resource_upper_bound(threads,registers,shared,props,max_blocks=24):
    if threads<1:raise ValueError('empty CTA')
    limits={'CTA_arch_limit':max_blocks,'thread_limit':props['max_threads_per_multi_processor']//threads}
    if registers:limits['register_limit_no_rounding']=props['regs_per_multiprocessor']//(threads*registers)
    if shared:limits['shared_limit']=props['shared_memory_per_multiprocessor']//shared
    ctas=min(limits.values());warps=math.ceil(threads/props['warp_size'])
    return dict(CTA_upper_bound=ctas,warp_occupancy_upper_bound=min(1.,ctas*warps/(props['max_threads_per_multi_processor']/props['warp_size'])),
        limiting_resources=[k for k,v in limits.items() if v==ctas],limits=limits,
        measured_occupancy=False,register_allocation_rounding_and_subpartition_limits_not_modeled=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--sqlite',type=Path,required=True);p.add_argument('--hardware',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    hardware=json.loads(args.hardware.read_text());props=hardware['gpu_properties']
    if props['major']!=8 or props['minor']!=9:raise ValueError('this resource-cap model is explicitly Ada SM89 only')
    db=sqlite3.connect(f'file:{args.sqlite.resolve()}?mode=ro',uri=True)
    strings=dict(db.execute('SELECT id,value FROM StringIds'))
    scopes=[]
    for a,b,t,i in db.execute('SELECT start,end,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL'):
        name=t if t is not None else strings.get(i,'')
        if name.startswith('fullflow/'):scopes.append((a,b))
    if not scopes:raise ValueError('full-flow range absent')
    start=min(a for a,b in scopes);end=max(b for a,b in scopes)
    query='''SELECT demangledName,gridX,gridY,gridZ,blockX,blockY,blockZ,registersPerThread,
    staticSharedMemory,dynamicSharedMemory,localMemoryPerThread,count(*),sum(end-start)
    FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=?
    GROUP BY demangledName,gridX,gridY,gridZ,blockX,blockY,blockZ,registersPerThread,
    staticSharedMemory,dynamicSharedMemory,localMemoryPerThread ORDER BY sum(end-start) DESC'''
    rows=[]
    for n,gx,gy,gz,bx,by,bz,r,ss,ds,local,count,ns in db.execute(query,(start,end)):
        threads=bx*by*bz;shared=ss+ds;grid=gx*gy*gz
        rows.append(dict(kernel=strings[n],grid=[gx,gy,gz],block=[bx,by,bz],threads_per_CTA=threads,
            registers_per_thread=r,static_shared_bytes=ss,dynamic_shared_bytes=ds,
            local_memory_per_thread_bytes=local,local_memory_is_not_measured_spill_traffic=True,
            calls=count,GPU_service_sum_s=ns/1e9,grid_CTAs=grid,waves_at_one_CTA_per_SM=grid/props['multi_processor_count'],
            resource_bounds=resource_upper_bound(threads,r,shared,props)))
    db.close()
    with args.sqlite.open('rb') as handle:sha=hashlib.file_digest(handle,'sha256').hexdigest()
    report=dict(status='pass',source=str(args.sqlite.resolve()),source_sha256=sha,hardware=str(args.hardware.resolve()),
                CPU_CUDA_launch_metadata_not_Ncu_counters=True,rows=rows)
    with args.output.open('x') as handle:json.dump(report,handle,indent=2);handle.write('\n')
    print(json.dumps(dict(status='pass',kernel_launch_variants=len(rows))))


if __name__=='__main__':main()
