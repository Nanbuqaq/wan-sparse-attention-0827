#!/usr/bin/env python3
"""Audit two-device CUPTI activity, with a real Perfetto-compatible trace.

Device activity is not SM occupancy; provisioned GPU-seconds are not measured
energy or GPU compute service. No inference or mutation of the source DB occurs.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_generator_timeline import duration,intersection,require_node_graph_trace
from scripts.audit_full_flow_timeline import summarize
from scripts.export_mentor_perfetto import write_trace


def analyze(db,summary):
    if summary.get('status')!='pass' or not summary.get('observer_noise_latent_RGB_equivalence'):
        raise ValueError('actual reference equivalence must pass before trace promotion')
    if summary.get('physical_GPU_mapping')!='0,1':
        raise ValueError('this audited deployment requires the locked 0,1 pair')
    require_node_graph_trace(db)
    strings=dict(db.execute('SELECT id,value FROM StringIds'))
    scopes=[]
    for a,b,text,textid,tid in db.execute('SELECT start,end,text,textId,globalTid FROM NVTX_EVENTS WHERE end>start'):
        name=text if text is not None else strings.get(textid,'')
        if name.startswith('native_pipeline/'):
            scopes.append((a,b,name,tid))
    windows=[r for r in scopes if r[2]=='native_pipeline/full_run']
    if len(windows)!=1:raise ValueError('exactly one completed native_pipeline/full_run marker required')
    start,end,_,_=windows[0]
    activities=[]
    for a,b,dev,pid,stream,name in db.execute(
        'SELECT start,end,deviceId,globalPid,streamId,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start<? AND end>?',(end,start)):
        activities.append(dict(a=max(a,start),b=min(b,end),device=dev,pid=pid,stream=stream,
                               kind='kernel',size=0,name=strings.get(name,str(name))))
    devices={r['device'] for r in activities};pids={r['pid'] for r in activities}
    if devices!={0,1} or len(pids)!=1:
        raise ValueError(f'expected both device kernel streams in one process, got devices={devices}, pids={pids}')
    for a,b,dev,pid,stream,size,kind in db.execute(
        'SELECT start,end,deviceId,globalPid,streamId,bytes,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start<? AND end>?',(end,start)):
        if dev not in devices or pid not in pids:raise ValueError('copy process/device outside the paired inference')
        label={1:'H2D',2:'D2H',8:'D2D'}.get(kind,'other_copy')
        activities.append(dict(a=max(a,start),b=min(b,end),device=dev,pid=pid,stream=stream,
                               kind=label,size=size,name=label))
    tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if 'CUPTI_ACTIVITY_KIND_MEMSET' in tables:
        for a,b,dev,pid,stream,size in db.execute(
            'SELECT start,end,deviceId,globalPid,streamId,bytes FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE start<? AND end>?',(end,start)):
            if dev not in devices or pid not in pids:raise ValueError('memset process/device mismatch')
            activities.append(dict(a=max(a,start),b=min(b,end),device=dev,pid=pid,stream=stream,
                                   kind='memset',size=size,name='memset'))
    by_device={dev:[r for r in activities if r['device']==dev] for dev in (0,1)}
    metric={dev:summarize([(r['a'],r['b'],r['kind'],r['size']) for r in rows]) for dev,rows in by_device.items()}
    kernel_spans={dev:[(r['a'],r['b']) for r in rows if r['kind']=='kernel'] for dev,rows in by_device.items()}
    concurrent=intersection(kernel_spans[0],kernel_spans[1])/1e9
    ledger=summary['video_pipeline']
    required={(0,'D2H'):ledger['latent_D2H_bytes'],(1,'H2D'):ledger['latent_H2D_bytes'],
              (1,'D2H'):ledger['pixel_D2H_bytes']}
    for (device,kind),minimum in required.items():
        if metric[device][kind]['bytes']<minimum:
            raise ValueError(f'CUPTI traffic incomplete: device={device} {kind} < recorded payload {minimum}')
    wall=(end-start)/1e9
    encode_scopes=[(max(a,start),min(b,end)) for a,b,name,tid in scopes
                   if name=='native_pipeline/encode' and min(b,end)>max(a,start)]
    encode_tids={tid for a,b,name,tid in scopes if name=='native_pipeline/encode'}
    decode_tids={tid for a,b,name,tid in scopes if name=='native_pipeline/VAE_decode_group'}
    encode_union=duration(encode_scopes)/1e9
    encode_overlap={device:intersection(encode_scopes,kernel_spans[device])/1e9 for device in (0,1)}
    result=dict(status='pass',scope='one_equivalent_native_two_gpu_diagnostic_not_repeated_speedup',
        parent_wall_s=wall,per_device=metric,cross_device_kernel_overlap_s=concurrent,
        GPU_overlap_proven=concurrent>0,any_device_activity_union_s=duration([(r['a'],r['b']) for r in activities])/1e9,
        profile_not_production_speedup=True,activity_fraction_is_not_SM_utilization=True,
        source_GPU='generator-side including control kernels',target_GPU='VAE and pixel preparation',
        CPU_output=dict(mode=ledger.get('encode_mode','legacy_inline'),NVTX_ranges=len(encode_scopes),
            encode_host_union_s=encode_union,overlap_GPU0_kernel_s=encode_overlap[0],overlap_GPU1_kernel_s=encode_overlap[1],
            encode_fraction_overlapping_GPU1_kernel=encode_overlap[1]/encode_union if encode_union else None,
            encode_and_decode_CPU_threads_distinct=bool(encode_tids and decode_tids and encode_tids.isdisjoint(decode_tids)),
            includes_pixel_conversion_hash_encode_mux=True,not_end_to_end_saved_time=True),
        recorded_minimum_payloads=[dict(device=d,direction=k,bytes=v) for (d,k),v in required.items()])
    return result,dict(start=start,end=end,scopes=scopes,activities=activities)


def trace_events(raw):
    def meta(pid,tid,name,value):return dict(ph='M',pid=pid,tid=tid,name=name,args={'name':value})
    events=[meta(10,0,'process_name','CPU native pipeline scopes')]
    tids={tid:i+1 for i,tid in enumerate(sorted({r[3] for r in raw['scopes']}))}
    for tid,mapped in tids.items():events.append(meta(10,mapped,'thread_name',f'CPU thread {tid}'))
    for dev in (0,1):
        events.append(meta(100+dev,0,'process_name',f'GPU {dev} · '+('generator' if dev==0 else 'VAE/pixels')))
        for stream in sorted({r['stream'] for r in raw['activities'] if r['device']==dev}):
            events.append(meta(100+dev,stream,'thread_name',f'CUDA stream {stream}'))
    begin,end=raw['start'],raw['end']
    for a,b,name,tid in raw['scopes']:
        a,b=max(a,begin),min(b,end)
        if b>a:events.append(dict(ph='X',name=name,pid=10,tid=tids[tid],ts=(a-begin)/1000,dur=(b-a)/1000,cat='CPU NVTX'))
    for row in raw['activities']:
        events.append(dict(ph='X',name=row['name'],pid=100+row['device'],tid=row['stream'],
            ts=(row['a']-begin)/1000,dur=(row['b']-row['a'])/1000,cat=row['kind'],
            args=dict(device=row['device'],bytes=row['size'],actual_CUPTI_activity=True)))
    return sorted(events,key=lambda r:r.get('ts',-1))


def main():
    p=argparse.ArgumentParser();p.add_argument('--sqlite',type=Path,required=True);p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:
        with sqlite3.connect(args.sqlite.resolve().as_uri()+'?mode=ro',uri=True) as db:
            result,raw=analyze(db,json.loads(args.summary.read_text()))
        for name,path in [('sqlite',args.sqlite),('summary',args.summary),('analysis_script',Path(__file__))]:
            with path.open('rb') as handle:result[name+'_sha256']=hashlib.file_digest(handle,'sha256').hexdigest()
        result['source_sqlite']=str(args.sqlite.resolve());result['source_summary']=str(args.summary.resolve())
        events=trace_events(raw)
        write_trace(args.output/'native_two_gpu.perfetto.json.gz',events,dict(scope=result['scope'],
            trace_is_actual_CUPTI=True,cross_device_kernel_overlap_s=result['cross_device_kernel_overlap_s']))
        result['exported_events']=len(events)
    except Exception:
        result=dict(status='fail',traceback=traceback.format_exc())
        raise
    finally:
        (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
