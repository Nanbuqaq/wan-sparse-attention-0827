#!/usr/bin/env python3
"""Two-slot D2H/pageable-commit overlap pilot with real resident Attention.

Includes CPU commit and allocation, but excludes prototype indexing/model/VAE.
It is not integrated end-to-end overlap. CUDA event spans require Nsight audit.
"""
from __future__ import annotations
import argparse
import json
import os
import hashlib
import subprocess
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.offload import ArchiveOffloadStager
from adapters.longlive_sparse.staging import PinnedStagingPool
from adapters.longlive_sparse.route_plan import HistoryRoutePlan,map_union_coordinates
from adapters.longlive_sparse.profiling import nvtx_range


def overlap_ms(first,second):
    def merge(items):
        out=[]
        for start,end in sorted(items):
            if out and start<=out[-1][1]:
                out[-1]=(out[-1][0],max(end,out[-1][1]))
            else:
                out.append((start,end))
        return out
    return sum(max(0.,min(b,d)-max(a,c)) for a,b in merge(first) for c,d in merge(second))


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--repeats',type=int,default=30)
    parser.add_argument('--compute-stream',choices=('default','dedicated'),default='default')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    import flash_attn
    c=torch.load(args.capture,map_location='cpu',weights_only=True)
    plan=HistoryRoutePlan.from_state_dict(c['route_plan'])
    if plan.groups!=1 or not bool((plan.group_history_counts==plan.union_frame_ids.shape[-1]).all()):
        raise ValueError('pilot requires identical full-union consumption per head')
    indices=map_union_coordinates(plan,c['frame_ids'],c['token_ids']).cuda()
    q=c['query'].cuda()
    def selected(name):
        tensor=c[name].cuda().permute(0,2,1,3)
        return tensor.gather(2,indices[...,None].expand(-1,-1,-1,tensor.shape[-1])).permute(0,2,1,3)
    k=torch.cat((c['exact_key'].cuda(),selected('key')),1).contiguous()
    v=torch.cat((c['exact_value'].cuda(),selected('value')),1).contiguous()
    if c['key_unrotated'].shape[1]<9360:
        raise ValueError('six-frame historical capture required')
    raw_k,raw_v=c['key_unrotated'][:,:9360].cuda(),c['value'][:,:9360].cuda()
    sources=[(raw_k[:,offset:offset+4680].view(1,3,1560,q.shape[2],q.shape[3]),
              raw_v[:,offset:offset+4680].view(1,3,1560,q.shape[2],q.shape[3])) for offset in (0,4680)]
    expected=[(a.cpu(),b.cpu()) for a,b in sources]
    pool=PinnedStagingPool(slots=2,budget_bytes=128*1024**2,pin_memory=True)
    stager=ArchiveOffloadStager(pool)
    compute_stream=torch.cuda.Stream() if args.compute_stream=='dedicated' else torch.cuda.current_stream()
    reference=flash_attn.flash_attn_func(q,k,v,causal=False)
    torch.cuda.synchronize()
    def trial(mode):
        torch.cuda.synchronize()
        origin=torch.cuda.Event(enable_timing=True)
        origin.record()
        started=time.perf_counter()
        tickets=[]
        intervals=[]
        outputs=[]
        committed=[]
        def compute():
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(origin)
                begin.record()
                outputs.append(flash_attn.flash_attn_func(q,k,v,causal=False))
                end.record()
            intervals.append((begin,end))
        if mode=='serial':
            for key,value in sources:
                ticket=stager.launch(key,value)
                tickets.append(ticket)
                committed.append(stager.complete(ticket))
                compute()
                if args.compute_stream=='dedicated':
                    intervals[-1][1].synchronize()
        else:
            for key,value in sources:
                tickets.append(stager.launch(key,value))
            compute()
            compute()
            committed=[stager.complete(ticket) for ticket in tickets]
        torch.cuda.synchronize()
        wall=time.perf_counter()-started
        copies=[(origin.elapsed_time(t.start_event),origin.elapsed_time(t.ready_event)) for t in tickets]
        compute_ranges=[(origin.elapsed_time(a),origin.elapsed_time(b)) for a,b in intervals]
        assert all(torch.equal(reference,out) for out in outputs)
        for (key,value,metrics),(refk,refv) in zip(committed,expected):
            assert torch.equal(key,refk) and torch.equal(value,refv) and not key.is_pinned()
        return {'wall_s':wall,'copy_event_spans_ms':copies,'compute_event_spans_ms':compute_ranges,
            'event_span_overlap_ms':overlap_ms(copies,compute_ranges),
            'cpu_commit_s':sum(item[2]['cpu_commit_s'] for item in committed),
            'payload_bytes':sum(item[2]['payload_bytes'] for item in committed),
            'correctness':'exact','torch_copy_calls':4}
    first={mode:trial(mode) for mode in ('serial','overlap')}
    for _ in range(5):
        trial('serial');trial('overlap')
    samples={mode:[] for mode in ('serial','overlap')}
    for repeat in range(args.repeats):
        for mode in (('serial','overlap') if repeat%2==0 else ('overlap','serial')):
            samples[mode].append(trial(mode))
    if args.profile:
        os.environ['LONGLIVE_NVTX']='1'
        torch.cuda.cudart().cudaProfilerStart()
        for mode in ('serial','overlap'):
            with nvtx_range('offload_pilot/'+mode):
                trial(mode)
        torch.cuda.cudart().cudaProfilerStop()
    result={'status':'pass','scope':'resident_attention_plus_D2H_and_pageable_CPU_commit_pilot',
        'source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'compute_stream':args.compute_stream,
        'cuda_device_max_connections':os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS'),
        'cuda_device_max_copy_connections':os.environ.get('CUDA_DEVICE_MAX_COPY_CONNECTIONS'),
        'capture':args.capture,'capture_sha256':hashlib.sha256(Path(args.capture).read_bytes()).hexdigest(),
        'runtime_overlap_integrated':False,'prototype_indexing_and_model_cost_included':False,
        'same_route_sha':plan.digest(),'gpu':torch.cuda.get_device_name(),'warmup':5,'repeats':args.repeats,
        'first_use':first,'samples':samples,'pool':pool.as_dict(),
        'summary':{mode:{'median_s':float(np.median([row['wall_s'] for row in rows])),
            'p95_s':float(np.percentile([row['wall_s'] for row in rows],95)),
            'median_event_span_overlap_ms':float(np.median([row['event_span_overlap_ms'] for row in rows]))}
            for mode,rows in samples.items()},
        'overlap_evidence_limit':'event spans only until Nsight memcpy/kernel interval audit',
        'end_to_end_speed_claim':False}
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':
    main()
