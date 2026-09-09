#!/usr/bin/env python3
"""Source-linked service-time comparison; setup is explicitly not amortized away."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    p=argparse.ArgumentParser();p.add_argument('--measurements',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);d=json.loads(args.measurements.read_text())
    assert d['status']=='pass' and len(d['rows'])==19
    assert len({r['logical_coordinate_sha256'] for r in d['rows']})==1
    assert all(r['all_selected_KV_bitwise_equal'] and r['measurement_count']==30 for r in d['rows'])
    chosen=[r for r in d['rows'] if r['mode']=='pre_pinned_runs' or (r['layout']=='exact' and r['mode'] in ('staged_separate','prepacked_exact'))]
    chosen.sort(key=lambda r:r['metrics']['wall_ms']['median'])
    labels=[r['layout']+' / '+r['mode'] for r in chosen]
    med=np.array([r['metrics']['wall_ms']['median'] for r in chosen]);p95=np.array([r['metrics']['wall_ms']['p95'] for r in chosen])
    fig,ax=plt.subplots(figsize=(11,5));y=np.arange(len(chosen))
    ax.barh(y,med,color=['#2d8c7c' if r['mode']=='prepacked_exact' else '#5485ad' for r in chosen])
    ax.scatter(p95,y,marker='|',s=160,color='#b84646',label='p95')
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel('Measured demand wall time (ms)');ax.grid(axis='x',alpha=.2)
    ax.set_title('Same logical ROI, same selected K/V | RTX 4090 | actual single-layer replay')
    for i,r in enumerate(chosen):ax.text(med[i]+.04,i,f'{med[i]:.3f} ms; {r["H2D_bytes"]/1e6:.2f} MB; {r["copy_calls"]} copies',va='center',fontsize=8)
    ax.set_xlim(0,max(p95)*1.5);ax.legend();fig.tight_layout();fig.savefig(args.output/'demand_cost.png',dpi=150);plt.close(fig)
    lines=['# 同一区域，不同组织与复用策略','',
        '19点全部通过每次K/V逐位验证；5轮warmup，30轮随机交错测量。v1保留，以下使用补齐固定ROI缓存并消除identity gather的v2。','',
        '| 路径 | H2D MB | copy数 | demand median / p95 ms |','| --- | ---: | ---: | ---: |']
    for r in chosen:lines.append(f'| {r["layout"]}/{r["mode"]} | {r["H2D_bytes"]/1e6:.3f} | {r["copy_calls"]} | {r["metrics"]["wall_ms"]["median"]:.3f} / {r["metrics"]["wall_ms"]["p95"]:.3f} |')
    lines+=['','固定ROI缓存最快，但要求source storage和selected坐标不变；换route必须重建，不能当自由动态布局。',
        f'其一次pack/pin创建观察为{d["fixed_route_packed_cache"]["pack_pin_creation_ms"]:.3f}ms，pinned缓存仅{d["fixed_route_packed_cache"]["pinned_bytes"]/1e6:.3f}MB。',
        'spatial4的完整archive构建与pin需另付成本，不能只拿warm0.889ms对比需求打包。',
        '源ROI只占9.73%，exact却有206个run copy；spatial4多搬一倍字节但66个copy，说明copy粒度和次数也重要。',
        'K/V分别pack的路径优于本实现的一次fused pack；减少API次数不保证降低完整取回时间。','',
        '解释：先问选择是否能复用，再选择预打包缓存或支持动态区域的物理布局。固定ROI不应该强制使用大page。',
        '范围：一个实际层、固定过去颜色ROI；不是全30层视频加速，不是在线分组方法，不是已验证overlap。',
        '布局构建/pin只有单次创建观察且两轮有波动，不给确定摊销门槛。CPU共享环境和allocator reserved未完整隔离/测量。','']
    (args.output/'INTERPRETATION.md').write_text('\n'.join(lines))
    (args.output/'source_audit.json').write_text(json.dumps(dict(status='pass',source=str(args.measurements),sha256=hashlib.sha256(args.measurements.read_bytes()).hexdigest(),
        configurations=19,measurement_replays=570,warmup_replays=95,all_KV_exact=True,not_independent_video_samples=True),indent=2)+'\n')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>同ROI的组织与复用</title><style>body{font:17px system-ui;max-width:1200px;margin:28px auto;padding:0 20px}img{max-width:100%}</style><h1>先决定什么能复用，再选择怎么搬</h1><p>同一逻辑ROI，19个配置逐次验证K/V相同。计入demand pack/pin/H2D/GPU整理；布局建立与固定ROI缓存创建单独报告。不是视频加速，也不是overlap证明。</p><img src="demand_cost.png"><p>固定ROI预打包约0.375ms；动态布局中的spatial4预pin约0.889ms。两者前提不同，构建成本不能忽略。</p><a href="INTERPRETATION.md">完整表格与边界</a> · <a href="source_audit.json">来源SHA</a>')
    print(json.dumps(dict(status='pass',configurations=19,output=str(args.output))))


if __name__=='__main__':main()
