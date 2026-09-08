#!/usr/bin/env python3
"""Source-locked memory motivation and measured restoration tradeoff figures."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import av
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont
from PIL import Image,ImageDraw,ImageFont


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);sources=[];figures=[]
    def load(path):
        raw=path.read_bytes();sources.append(dict(path=str(path.resolve()),sha256=hashlib.sha256(raw).hexdigest()))
        return json.loads(raw)
    video_root=args.results/'videos/sprint24h_20260907/longlive2_native_cut509_h_v1'
    canvas=Image.new('RGB',(1080,1030),'white');draw=ImageDraw.Draw(canvas)
    font_path=findfont('DejaVu Sans')
    font=ImageFont.truetype(font_path,20)
    small=ImageFont.truetype(font_path,16)
    for i,label in enumerate(('Before absence: generated information','Target-absent interval','Return: native output')):
        draw.text((15+i*355,12),label,fill='#263345',font=small)
    for lane in range(4):
        root=video_root/f'lane{lane}';d=load(root/'summary.json');video=root/'video.mp4'
        captures={};wanted={188,300,508}
        with av.open(str(video)) as container:
            for index,frame in enumerate(container.decode(video=0)):
                if index in wanted:captures[index]=frame.to_image()
        if set(captures)!=wanted:raise ValueError('native figure video incomplete')
        label=('Generated toy identity' if lane<2 else 'Achieved red-bead state')+' / seed '+str(d['seed'])
        y=52+lane*239;draw.text((15,y),label,fill='#263345',font=font)
        for col,index in enumerate((188,300,508)):
            picture=captures[index];picture.thumbnail((345,190));canvas.paste(picture,(15+col*355,y+32))
        sources.append(dict(path=str(video.resolve()),sha256=hashlib.sha256(video.read_bytes()).hexdigest()))
    canvas.save(args.output/'native_memory_failure.png')
    figures.append(dict(name='native_memory_failure',extension='png',caption='实际H800原生5B、两任务双seed、正确分镜。目标形成→长于有效近期上下文的真实离开→返回后身份/状态未保住。这里只是开发集motivation，不是新方法胜出。图中3帧为示例，完整128帧离开区间另已逐帧接触表审查。'))
    base=args.results/'metrics/sprint24h_20260907'
    benchmark=load(base/'native_restore_benchmark_v3_recipe/summary.json')
    if benchmark['status']!='pass' or not benchmark['full_original_KV_hash_gates']:raise ValueError('unverified timing')
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    state_mb=[benchmark['raw_KV_bytes']/1e6,benchmark['log_tensor_bytes']/1e6]
    axes[0].bar(['Materialized KV','Clean log'],state_mb,color=['#64748b','#0284c7']);axes[0].set_yscale('log')
    axes[0].set_ylabel('State tensor bytes (MB, log scale)');axes[0].set_title('Same positive KV state, different stored form')
    for x,value in enumerate(state_mb):axes[0].text(x,value*1.15,f'{value:,.1f} MB',ha='center')
    names=['raw_pageable','raw_bounded_pinned','clean_log_replay'];labels=['Raw pageable','Raw + bounded\npinned staging','Clean-log replay']
    med=[benchmark['medians_s'][k]*1000 for k in names];p95=[benchmark['p95_s'][k]*1000 for k in names]
    axes[1].bar(labels,med,color=['#64748b','#94a3b8','#0284c7'])
    axes[1].scatter(range(3),p95,color='#dc2626',marker='_',s=150,label='p95');axes[1].legend()
    axes[1].set_ylabel('Warm restore wall time (ms)');axes[1].set_title('Lower state bytes are not a latency win here')
    fig.tight_layout();fig.savefig(args.output/'native_restore_tradeoff.svg',bbox_inches='tight')
    fig.savefig(args.output/'native_restore_tradeoff.png',dpi=160,bbox_inches='tight');plt.close(fig)
    figures.append(dict(name='native_restore_tradeoff',extension='svg',caption='RTX4090、32-frame正向KV状态；5次预热后30次随机交错测量。原始搬运更快，日志更小。约105MiB预分配pinned总预算，staging模式含CPU pack。模型权重共同且不计入state bytes；不是进程RSS或完整视频速度。此轮通过离线KV哈希找回兼容的16-warps数值配置，不能冒充无需核对数据的冷恢复。'))
    report=dict(status='pass',sources=sources,figures=figures,restore_state_MB=state_mb,restore_median_ms=med,
        restore_p95_ms=p95,semantic_quality_requires_separate_reviews=True)
    (args.output/'figure_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    sections=''.join('<section><h2>'+html.escape(f['name'].replace('_',' '))+'</h2><img src="'+f['name']+'.'+f['extension']+'"><p>'+html.escape(f['caption'])+'</p></section>' for f in figures)
    (args.output/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Native memory evidence</title><style>body{font:16px/1.7 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#263345}img{width:100%;height:auto}section{margin:40px 0}p{color:#475569}</style><h1>生成信息的记忆缺口，与重建式存储的取舍</h1>'+sections+'<p><a href="figure_manifest.json">数据及来源SHA</a></p></html>')
    print(json.dumps(dict(status='pass',figures=2,output=str(args.output))))


if __name__=='__main__':main()
