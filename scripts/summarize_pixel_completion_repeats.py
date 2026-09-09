#!/usr/bin/env python3
"""Three paired full-run repeats, no confidence interval or cross-workload claim."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import traceback


def summarize_pairs(audits):
    if len(audits)!=3:raise ValueError('exactly three preregistered pairs required')
    pairs=[]
    for repeat,audit in enumerate(audits):
        if audit['status']!='pass':raise ValueError(f'pair {repeat} failed; no successful-only speed summary')
        rows={r['mode']:r for r in audit['rows']}
        if set(rows)!={'inline','thread'} or len(audit['rows'])!=2:raise ValueError('both unique modes required')
        for row in rows.values():
            if row['status']!='pass' or not row['actual_full_latent_and_decoded_RGB_exact'] or row['frames']!=509:
                raise ValueError('full actual output equality required')
            if not row['pixel_slots_not_reused_before_sink_completion']:raise ValueError('unsafe pixel slot')
            if not math.isfinite(row['complete_s']) or row['complete_s']<=0:raise ValueError('invalid wall')
        a,b=rows['inline']['complete_s'],rows['thread']['complete_s']
        pairs.append(dict(repeat=repeat,inline_s=a,thread_s=b,latency_reduction=1-b/a,speedup=a/b,
            inline_first_packet_s=rows['inline']['first_packet_s'],thread_first_packet_s=rows['thread']['first_packet_s'],
            inline_producer_backpressure_s=rows['inline']['producer_backpressure_s'],
            thread_producer_backpressure_s=rows['thread']['producer_backpressure_s']))
    modes={}
    for mode in ('inline','thread'):
        times=[r[mode+'_s'] for r in pairs];first=[r[mode+'_first_packet_s'] for r in pairs]
        modes[mode]=dict(n=3,median_complete_s=statistics.median(times),minimum_s=min(times),maximum_s=max(times),
            median_first_packet_s=statistics.median(first),first_packet_min_s=min(first),first_packet_max_s=max(first))
    return dict(status='pass',pairs=pairs,modes=modes,median_pair_latency_reduction=statistics.median(r['latency_reduction'] for r in pairs),
        minimum_pair_latency_reduction=min(r['latency_reduction'] for r in pairs),
        all_three_pairs_exceed_10percent=all(r['latency_reduction']>=.1 for r in pairs),
        no_confidence_interval_claim=True,one_workload_one_hardware_pair_only=True,
        no_one_GPU_throughput_or_new_quality_claim=True,loading_excluded_but_pipeline_construction_included=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);report=dict(status='running',audits=[])
    try:
        audits=[]
        for i in range(3):
            path=args.root/f'repeat{i}.json';audit=json.loads(path.read_text());audits.append(audit)
            report['audits'].append(dict(repeat=i,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),status=audit['status']))
        report.update(summarize_pairs(audits))
        table=''.join(f'<tr><td>{r["repeat"]}</td><td>{r["inline_s"]:.3f}</td><td>{r["thread_s"]:.3f}</td><td>{100*r["latency_reduction"]:.2f}%</td></tr>' for r in report['pairs'])
        (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Native pixel completion repeats</title>'
            '<style>body{font:18px system-ui;max-width:950px;margin:35px auto}td,th{padding:10px 24px;border-bottom:1px solid #ddd}</style>'
            '<h1>同双卡、同完整输出：三组配对重复</h1><p>模型加载不在计时中；pipeline初始化、生成、解码和完整交付在计时中。'
            '仅一条509帧工作负载，不是总体置信区间或单卡吞吐优势。增加一个CPU输出工作线程及43.254MB pinned输出缓冲。</p>'
            '<table><tr><th>重复</th><th>Inline秒</th><th>线程秒</th><th>延迟下降</th></tr>'+table+'</table><p><a href="summary.json">全部审计与首包数据</a></p>')
    except Exception:report.update(status='fail',traceback=traceback.format_exc());raise
    finally:
        (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
