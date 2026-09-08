#!/usr/bin/env python3
"""Nsight SQLite -> real Chrome JSON traces accepted by Perfetto.

Overview counters are traced activity fractions, NOT SM occupancy. Detail keeps
actual CUDA API/kernel/copy spans and launch-correlation flows. CPU NVTX ending
does not imply GPU completion. Original sources stay read-only.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.audit_full_flow_timeline import attribute_runtime,MAJOR,summarize
from scripts.audit_generator_timeline import merge,require_node_graph_trace


def activity_bins(spans,start,end,width):
    bins=[0]*((end-start+width-1)//width)
    for a,b in merge(spans):
        a=max(a,start);b=min(b,end)
        if b<=a:continue
        first=(a-start)//width;last=(b-1-start)//width
        for i in range(first,last+1):
            bins[i]+=min(b,start+(i+1)*width)-max(a,start+i*width)
    return [(start+i*width,100*v/min(width,end-start-i*width)) for i,v in enumerate(bins)]


def write_trace(path,events,metadata):
    with gzip.open(path,'wt',encoding='utf-8') as handle:
        handle.write('{"displayTimeUnit":"ms","metadata":'+json.dumps(metadata,ensure_ascii=False)+',"traceEvents":[')
        for i,event in enumerate(events):
            if i:handle.write(',')
            json.dump(event,handle,ensure_ascii=False,separators=(',',':'))
        handle.write(']}\n')


def gpu_stage(name,leaf,major):
    flash=('flash' in name.lower() and ('fwd' in name.lower() or 'attn' in name.lower()))
    if flash and major=='generation.complete' and leaf not in ('transformer.cross_attn','text.encode_with_dynamic_swap'):
        return 'history_and_exact_attention_core'
    return leaf


def export(sqlite_path,output,report_path=None,generic=False,detail_window=None):
    output.mkdir(parents=True,exist_ok=False)
    db=sqlite3.connect(f'file:{sqlite_path.resolve()}?mode=ro',uri=True);require_node_graph_trace(db)
    strings=dict(db.execute('SELECT id,value FROM StringIds'))
    scopes=[]
    for a,b,text,text_id,tid in db.execute('SELECT start,end,text,textId,globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL'):
        name=text if text is not None else strings.get(text_id,'')
        if generic or name.startswith('fullflow/'):
            scopes.append(dict(start=a,end=b,name=name.removeprefix('fullflow/'),tid=tid,args={}))
    if not scopes:raise ValueError('no relevant NVTX scopes')
    report=json.loads(report_path.read_text()) if report_path else None
    metadata_matches={}
    if report:
        if report['status']!='pass' or not report['instrumented_latent_exact_control']:
            raise ValueError('profile must pass uninstrumented latent control')
        a_by_name=defaultdict(list);r_by_name=defaultdict(list)
        for s in scopes:a_by_name[s['name']].append(s)
        for r in report['trace']['records']:r_by_name[r['name']].append(r)
        for name,rows in a_by_name.items():
            refs=r_by_name[name];ok=len(rows)==len(refs);metadata_matches[name]=ok
            if ok:
                for s,r in zip(sorted(rows,key=lambda z:z['start']),sorted(refs,key=lambda z:z['start_s'])):
                    s['args'].update(r.get('metadata',{}));s['args']['profile_record_id']=r['id']
    start=min(r['start'] for r in scopes);end=max(r['end'] for r in scopes)
    runtimes=list(db.execute('SELECT correlationId,start,end,globalTid,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start>=? AND start<=?',(start,end)))
    runtime_map={row[0]:row for row in runtimes}
    assignments=attribute_runtime([(r['start'],r['end'],r['name'],r['tid']) for r in scopes],[(c,a,t) for c,a,b,t,n in runtimes])
    activities=[]
    for a,b,c,n,stream,pid,device in db.execute('SELECT start,end,correlationId,demangledName,streamId,globalPid,deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start<? AND end>?',(end,start)):
        name=strings.get(n,str(n));leaf,major=assignments.get(c,('unattributed','unattributed'))
        activities.append(dict(start=a,end=b,corr=c,name=name,kind='kernel',bytes=0,stream=stream,pid=pid,device=device,
                               leaf=leaf,major=major,stage=gpu_stage(name,leaf,major)))
    kinds={1:'H2D',2:'D2H',8:'D2D'}
    for a,b,c,size,kind,stream,pid,device in db.execute('SELECT start,end,correlationId,bytes,copyKind,streamId,globalPid,deviceId FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start<? AND end>?',(end,start)):
        leaf,major=assignments.get(c,('unattributed','unattributed'));label=kinds.get(kind,'other_copy')
        activities.append(dict(start=a,end=b,corr=c,name=label,kind=label,bytes=size,stream=stream,pid=pid,device=device,leaf=leaf,major=major,stage=leaf))
    tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if 'CUPTI_ACTIVITY_KIND_MEMSET' in tables:
        for a,b,c,size,stream,pid,device in db.execute('SELECT start,end,correlationId,bytes,streamId,globalPid,deviceId FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE start<? AND end>?',(end,start)):
            leaf,major=assignments.get(c,('unattributed','unattributed'))
            activities.append(dict(start=a,end=b,corr=c,name='memset',kind='memset',bytes=size,stream=stream,pid=pid,device=device,leaf=leaf,major=major,stage=leaf))
    if len({r['pid'] for r in activities})!=1 or len({r['device'] for r in activities})!=1:
        raise ValueError('export one CUDA process/device at a time')
    db.close()
    tids={t:i+1 for i,t in enumerate(sorted({r['tid'] for r in scopes}|{r[3] for r in runtimes}))}
    stream_ids={s:i+1 for i,s in enumerate(sorted({r['stream'] for r in activities}))}
    def meta_event(name,pid,tid,args):return dict(ph='M',name=name,pid=pid,tid=tid,args=args)
    metadata_events=[meta_event('process_name',10,0,{'name':'CPU · NVTX / CUDA launch'}),
        meta_event('process_name',20,0,{'name':'GPU · actual kernels and copies'}),
        meta_event('process_name',30,0,{'name':'GPU activity fractions · NOT SM utilization'})]
    metadata_events += [meta_event('thread_name',10,v,{'name':f'CPU thread {k}'}) for k,v in tids.items()]
    metadata_events += [meta_event('thread_name',20,v,{'name':f'CUDA stream {k}'}) for k,v in stream_ids.items()]
    def span_event(name,a,b,pid,tid,category,args,lo,hi):
        x,y=max(a,lo),min(b,hi)
        if y<=x:return None
        return dict(ph='X',name=name,cat=category,pid=pid,tid=tid,ts=(x-start)/1000,dur=(y-x)/1000,
                    args=dict(args,clipped_at_view_boundary=(a<lo or b>hi)))
    stage_data=defaultdict(list);major_data=defaultdict(list)
    for a in activities:
        entry=(a['start'],a['end'],a['kind'],a['bytes'])
        stage_data[(a['major'],a['stage'])].append(entry);major_data[a['major']].append(entry)
    quant=dict(major_stages={k:summarize(v) for k,v in major_data.items()},
        stages=[dict(major=k[0],stage=k[1],**summarize(v)) for k,v in sorted(stage_data.items())],
        device_memory_copies_not_all_DRAM_transactions=True,CPU_scope_is_not_GPU_completion=True,
        stage_assignment_uses_launch_correlation_not_GPU_time_containment=True,
        metadata_occurrence_matches=metadata_matches)
    forwards=[s for s in scopes if s['name']=='generator.forward' and s['args'].get('call_in_chunk')==0]
    selected=max(forwards,key=lambda z:z['args'].get('current_start_token',-1)) if forwards else None
    if selected:
        lo,hi=selected['start'],selected['end']
        owned=[a for a in activities if a['corr'] in runtime_map and lo<=runtime_map[a['corr']][1]<hi]
        hi=max([hi]+[r['end'] for r in owned]);lo=max(start,lo-100_000);hi=min(end,hi+100_000)
    else:lo,hi=start,end
    if detail_window is not None:
        lo=start+int(detail_window[0]*1e9);hi=start+int(detail_window[1]*1e9)
        if not start<=lo<hi<=end:raise ValueError('detail window lies outside captured ROI')
    common=dict(source_sqlite=str(sqlite_path.resolve()),actual_Nsight_CUPTI_and_NVTX=True,
                timestamp_unit='microseconds',original_origin_ns=start,shape_report=str(report_path) if report_path else None)
    def overview():
        yield from metadata_events
        for s in scopes:
            if generic or s['name'] in MAJOR|{'generator.forward','transformer.block'}:
                e=span_event(s['name'],s['start'],s['end'],10,tids[s['tid']],'CPU_scope',s['args'],start,end)
                if e:yield e
        for label,kinds_wanted in (('kernel active %',{'kernel'}),('H2D active %',{'H2D'}),('D2H active %',{'D2H'}),('D2D active %',{'D2D'}),('any traced GPU activity %',None)):
            spans=[(r['start'],r['end']) for r in activities if kinds_wanted is None or r['kind'] in kinds_wanted]
            for stamp,value in activity_bins(spans,start,end,5_000_000):
                yield dict(ph='C',name=label,pid=30,tid=0,ts=(stamp-start)/1000,args={'percent':value})
    def detail():
        yield from metadata_events
        selected_activities=[a for a in activities if a['start']<hi and a['end']>lo]
        for s in scopes:
            e=span_event(s['name'],s['start'],s['end'],10,tids[s['tid']],'CPU_scope',s['args'],lo,hi)
            if e:yield e
        for c,a,b,t,n in runtimes:
            # Waits, allocations and event APIs may emit no GPU activity but
            # are essential for diagnosing exposed host stalls.
            e=span_event(strings.get(n,str(n)),a,b,10,tids[t],'CUDA_API',{'correlation_id':c},lo,hi)
            if e:yield e
        flow=0
        for r in selected_activities:
            args={k:r[k] for k in ('corr','bytes','kind','leaf','major','stage')};args['full_kernel_name']=r['name']
            e=span_event(r['name'][:160],r['start'],r['end'],20,stream_ids[r['stream']],r['kind'],args,lo,hi)
            if e:yield e
            launch=runtime_map.get(r['corr'])
            if launch and lo<=launch[1]<=r['start']<hi:
                flow+=1
                yield dict(ph='s',cat='launch_correlation',name='CPU launch → GPU work',id=flow,pid=10,tid=tids[launch[3]],ts=(launch[1]-start)/1000)
                yield dict(ph='f',cat='launch_correlation',name='CPU launch → GPU work',id=flow,pid=20,tid=stream_ids[r['stream']],ts=(r['start']-start)/1000,bp='e')
    write_trace(output/'overview.trace.json.gz',overview(),dict(common,view='overview',counters_are_5ms_activity_fraction_not_occupancy=True))
    write_trace(output/'detail.trace.json.gz',detail(),dict(common,view='detail',window_relative_s=[(lo-start)/1e9,(hi-start)/1e9]))
    with sqlite_path.open('rb') as handle:source_sha=hashlib.file_digest(handle,'sha256').hexdigest()
    manifest=dict(status='pass',source_sqlite=str(sqlite_path.resolve()),source_sha256=source_sha,
        profile_status=report.get('status') if report else None,latent_control_exact=report.get('instrumented_latent_exact_control') if report else None,
        total_CPU_scopes=len(scopes),GPU_activities_in_ROI=len(activities),duration_s=(end-start)/1e9,
        detail_window_relative_s=[(lo-start)/1e9,(hi-start)/1e9],detail_forward_metadata=selected['args'] if selected else None,
        quantitative=quant,service_sums_not_critical_path=True,SM_occupancy_not_measured=True,
        detail_includes_all_CUDA_runtime_APIs_in_window=True,
        files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in output.glob('*.gz')})
    (output/'summary.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='quantitative'}))


def main():
    p=argparse.ArgumentParser();p.add_argument('--sqlite',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report',type=Path);p.add_argument('--generic',action='store_true')
    p.add_argument('--detail-window',type=float,nargs=2,metavar=('START_S','END_S'));args=p.parse_args()
    export(args.sqlite,args.output,args.report,args.generic,args.detail_window)


if __name__=='__main__':main()
