#!/usr/bin/env python3
"""One source-audited state-size/restore-latency plot, not a video-speed claim."""
import argparse
import hashlib
import html
import json
from pathlib import Path


def rows_from_report(data):
    if (data['status']!='pass' or not data['full_original_KV_hash_gates']
            or not data['recorded_kernel_recipes_applied'] or data['recipe_selected_with_offline_witness']):
        raise ValueError('recorded-recipe exact restoration required')
    if data['repeats']!=30 or data['warmup_per_mode']!=5:
        raise ValueError('this frozen figure requires five warmups and thirty blocked repeats')
    states={name:row['total_retained_state_tensor_bytes'] for name,row in data['hybrid_checkpoint_states'].items()}
    states.update(raw_pageable=data['raw_KV_bytes'],raw_bounded_pinned=data['raw_KV_bytes'],clean_log_replay=data['log_tensor_bytes'])
    if set(states)!=set(data['medians_s']):raise ValueError('unmatched timing/storage paths')
    labels={'raw_pageable':'Raw pageable','raw_bounded_pinned':'Raw + pinned staging','clean_log_replay':'Clean log'}
    result=[]
    for name,size in states.items():
        checkpoint=data['hybrid_checkpoint_states'].get(name)
        label=labels.get(name) or f"KV {checkpoint['committed_checkpoint_chunks']*8} frames + log tail"
        result.append(dict(path=name,label=label,state_tensor_bytes=size,median_s=data['medians_s'][name],p95_s=data['p95_s'][name]))
    return sorted(result,key=lambda r:(r['state_tensor_bytes'],r['median_s']))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();raw=args.report.read_bytes();data=json.loads(raw);rows=rows_from_report(data)
    args.output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,5))
    for i,row in enumerate(rows):
        x=row['state_tensor_bytes']/1e6;y=row['median_s']*1000;high=row['p95_s']*1000
        ax.vlines(x,y,high,color='#94a3b8',linewidth=2)
        ax.scatter([x],[y],color='#0284c7',s=55,zorder=3)
        ax.scatter([x],[high],color='#ef4444',marker='_',s=70)
        offset=(8,9) if 'pinned' not in row['path'] else (-8,8)
        ax.annotate(row['label'],(x,y),xytext=offset,textcoords='offset points',
                    ha='left' if offset[0]>0 else 'right',fontsize=9)
    ax.set_xscale('log');ax.set_xlim(min(r['state_tensor_bytes'] for r in rows)/1e6*.75,max(r['state_tensor_bytes'] for r in rows)/1e6*2)
    ax.set_xlabel('Retained state tensor bytes (MB, log scale; not process RSS)')
    ax.set_ylabel('Warm complete restore wall time (ms)')
    ax.set_title('Committed KV checkpoints + clean-log tails: exact-state tradeoff')
    ax.grid(alpha=.15);fig.tight_layout()
    fig.savefig(args.output/'tradeoff.png',dpi=160);fig.savefig(args.output/'tradeoff.svg');plt.close(fig)
    result=dict(status='pass',source=str(args.report.resolve()),source_sha256=hashlib.sha256(raw).hexdigest(),
                hardware=data['gpu'],source_commit=data['source_commit'],repeats=data['repeats'],rows=rows,
                snapshot_creation_and_eviction_not_in_restore_timing=True,not_video_or_process_RSS_gain=True)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    table='\n'.join(f"| {r['label']} | {r['state_tensor_bytes']/1e6:.3f} | {r['median_s']*1000:.2f} | {r['p95_s']*1000:.2f} |" for r in rows)
    (args.output/'RESULTS.md').write_text('# Exact hybrid restoration\n\n| Path | State MB | Median ms | p95 ms |\n|---|---:|---:|---:|\n'+table+'\n\nAll paths pass full original KV hashes. 5 warmup / 30 blocked randomized repeats.\nOnly restore wall time; checkpoint construction, model loading, eviction and cold storage are excluded.\nTensor-state bytes exclude model weights and recipe metadata, not process RSS.\n')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Hybrid native restore</title><style>body{max-width:1100px;margin:40px auto;font:16px/1.7 system-ui}img{width:100%}</style><h1>已提交KV检查点＋日志尾部</h1><img src="tradeoff.svg"><p>'+html.escape(data['gpu'])+'；全部原KV哈希通过；每路径5预热、30随机交错重复。红线为p95。</p><p>仅warm恢复，不含创建/驱逐/冷存储；不是完整视频加速或进程RSS收益。</p><a href="summary.json">数据与来源SHA</a>')
    print(json.dumps(result))


if __name__=='__main__':main()
