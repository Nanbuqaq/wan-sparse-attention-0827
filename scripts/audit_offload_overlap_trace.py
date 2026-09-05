#!/usr/bin/env python3
"""Audit actual Nsight D2H/FA2 intersections, not asynchronous API presence."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.benchmark_offload_overlap import overlap_ms


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sqlite',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    db=sqlite3.connect(args.sqlite)
    ranges=db.execute('select start,end,coalesce(text,(select value from StringIds where id=textId)) from NVTX_EVENTS where end is not null').fetchall()
    copies=db.execute('select start,end,streamId,bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=2 and bytes>1000000').fetchall()
    kernels=db.execute('select start,end,streamId from CUPTI_ACTIVITY_KIND_KERNEL where shortName in (select id from StringIds where value=?)',('flash_fwd_kernel',)).fetchall()
    result={}
    for start,end,label in ranges:
        if label not in ('offload_pilot/serial','offload_pilot/overlap'):
            continue
        local_copies=[row for row in copies if row[0]>=start and row[1]<=end]
        local_kernels=[row for row in kernels if row[0]>=start and row[1]<=end]
        result[label]={'d2h_bytes':sum(row[3] for row in local_copies),'d2h_copies':len(local_copies),
            'copy_streams':sorted({row[2] for row in local_copies}),
            'attention_streams':sorted({row[2] for row in local_kernels}),
            'attention_kernels':len(local_kernels),
            'd2h_service_ms':sum(row[1]-row[0] for row in local_copies)/1e6,
            'attention_service_ms':sum(row[1]-row[0] for row in local_kernels)/1e6,
            'actual_d2h_attention_overlap_ms':overlap_ms([row[:2] for row in local_copies],[row[:2] for row in local_kernels])/1e6}
    if len(result)!=2:
        raise RuntimeError('both pilot ranges are required')
    if result['offload_pilot/serial']['d2h_bytes']!=result['offload_pilot/overlap']['d2h_bytes']:
        raise RuntimeError('copy byte budgets differ')
    payload={'status':'pass','scope':'actual_GPU_event_intersections_not_video_speedup','ranges':result,
        'end_to_end_speed_claim':False,'nvtx_wall_not_used':'includes correctness auditing outside benchmark timer'}
    with Path(args.sqlite).open('rb') as handle:
        payload['sqlite_sha256']=hashlib.file_digest(handle,'sha256').hexdigest()
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
