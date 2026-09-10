#!/usr/bin/env python3
"""Attribute GPU service by CPU launch correlation, not wall-time overlap."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_generator_timeline import duration


def category(name):
    if name=='native/self_attention_core':return 'self_attention_wrapper'
    if name=='native/cross_attention_core':return 'cross_attention_wrapper'
    if name.startswith('native/self_attn/'):return 'self_QKV_norm_RoPE_window_output_projection'
    if name.startswith('native/cross_attn/'):return 'cross_attention_preparation_and_projection'
    if name.startswith('native/ffn/'):return 'FFN'


def scope_index(scopes):
    groups=defaultdict(list)
    for a,b,name,tid in scopes:
        cat=category(name)
        if cat:groups[(tid,cat)].append((a,b))
    return {key:(sorted(values),[a for a,b in sorted(values)]) for key,values in groups.items()}


def launch_category(index,tid,a,b):
    for cat in ('self_attention_wrapper','cross_attention_wrapper',
                'self_QKV_norm_RoPE_window_output_projection','cross_attention_preparation_and_projection','FFN'):
        ranges,starts=index.get((tid,cat),([],[]));i=bisect_right(starts,a)-1
        if i>=0 and ranges[i][1]>=b:return cat
    return 'unattributed_or_other_model_work'


def analyze(db):
    strings=dict(db.execute('select id,value from StringIds'))
    scopes=[(a,b,text or strings.get(textid,''),tid) for a,b,text,textid,tid in
            db.execute('select start,end,text,textId,globalTid from NVTX_EVENTS where end>start')]
    windows=[(a,b) for a,b,name,tid in scopes if name=='native_generation_only']
    if len(windows)!=1:raise ValueError('one complete native generation window required')
    begin,end=windows[0];index=scope_index(scopes)
    launches={}
    for a,b,tid,corr in db.execute('select start,end,globalTid,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME'):
        if corr in launches:raise ValueError('ambiguous CUDA runtime correlation ID')
        launches[corr]=(a,b,tid)
    service=defaultdict(float);counts=defaultdict(int);flash=defaultdict(float)
    kernel_spans=[];all_spans=[];unmatched=0;devices=set();processes=set()
    for a,b,dev,pid,name,corr in db.execute('select start,end,deviceId,globalPid,demangledName,correlationId from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=?',(begin,end)):
        devices.add(dev);processes.add(pid)
        if corr in launches:
            la,lb,tid=launches[corr];cat=launch_category(index,tid,la,lb)
        else:cat='missing_launch_correlation';unmatched+=1
        seconds=(b-a)/1e9;service[cat]+=seconds;counts[cat]+=1
        if 'flash_fwd_kernel' in strings[name]:flash[cat]+=seconds
        kernel_spans.append((a,b));all_spans.append((a,b))
    if len(devices)!=1 or len(processes)!=1:raise ValueError('single-process single-device generation expected')
    transfers=defaultdict(lambda:dict(bytes=0,service_s=0.,calls=0))
    for a,b,size,kind in db.execute('select start,end,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and end<=?',(begin,end)):
        label={1:'H2D',2:'D2H',8:'D2D'}.get(kind,'other')
        transfers[label]['bytes']+=size;transfers[label]['service_s']+=(b-a)/1e9;transfers[label]['calls']+=1
        all_spans.append((a,b))
    host_ranges=defaultdict(list)
    for a,b,name,tid in scopes:
        cat=category(name)
        if cat and min(b,end)>max(a,begin):host_ranges[cat].append((max(a,begin),min(b,end)))
    wall=(end-begin)/1e9
    return dict(status='pass',parent_window_s=wall,GPU_kernel_service_s=dict(service),
        GPU_kernel_calls=dict(counts),FA2_kernel_service_s=dict(flash),
        GPU_kernel_union_s=duration(kernel_spans)/1e9,GPU_kernel_and_copy_union_s=duration(all_spans)/1e9,
        outside_kernel_and_copy_s=wall-duration(all_spans)/1e9,transfers=dict(transfers),
        host_scope_union_s={k:duration(v)/1e9 for k,v in host_ranges.items()},
        missing_kernel_launch_correlations=unmatched,devices=sorted(devices),
        host_scope_time_is_not_pure_CPU_compute=True,nested_host_ranges_not_additive=True,
        activity_is_not_SM_utilization=True,no_hardware_HBM_counter=True,
        profiled_diagnostic_not_timing_repetition=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--sqlite',type=Path,required=True)
    p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    summary=json.loads(args.summary.read_text());assert summary['status']=='pass'
    with sqlite3.connect(args.sqlite.resolve().as_uri()+'?mode=ro',uri=True) as db:report=analyze(db)
    report['case']={k:summary.get(k) for k in ('seed','gpu','native_DiT_s','native_VAE_s','load_s','latent_sha256')}
    for name,path in [('sqlite',args.sqlite),('summary',args.summary),('script',Path(__file__))]:
        with path.open('rb') as h:report[name+'_sha256']=hashlib.file_digest(h,'sha256').hexdigest()
    (args.output/'profile.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
