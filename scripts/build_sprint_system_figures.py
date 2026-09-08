#!/usr/bin/env python3
"""Complete-pipeline factorial and bounded-page evidence, with source SHA locks."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);sources=[];figures=[];data=[]
    def load(relative):
        path=args.results/relative;raw=path.read_bytes()
        sources.append(dict(path=str(path.resolve()),sha256=hashlib.sha256(raw).hexdigest()))
        return json.loads(raw)
    def save(fig,name,caption):
        fig.tight_layout();fig.savefig(args.output/(name+'.svg'),bbox_inches='tight')
        fig.savefig(args.output/(name+'.png'),bbox_inches='tight',dpi=160);plt.close(fig)
        figures.append(dict(name=name,caption=caption))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    prefix='metrics/sprint24h_20260907/'
    factorial=load('videos/sprint24h_20260907/streaming_factorial477_h200_v1/factorial_audit.json')
    fig,axes=plt.subplots(1,2,figsize=(11,4.2));labels=[]
    contrasts=[('RoPE_under_batch','Direct RoPE alone','#6b7280'),
        ('streaming_under_upstream_RoPE','Streaming alone','#0284c7'),('combined','Both','#7c3aed')]
    for i,lane in enumerate(factorial['lanes']):
        if lane['gpu']!='NVIDIA H200' or not lane['same_noise_routes_latents_RGB']:raise ValueError('factorial hardware/equivalence gate failed')
        label=('Dense' if lane['method']=='rag_dense' else 'Final')+' / '+lane['prompt']['prompt_id'].removeprefix('calibration_')
        labels.append(label)
        for j,(key,name,color) in enumerate(contrasts):
            values=lane['contrasts'][key]['paired_speedups'];x=i+(j-1)*.23
            axes[0].scatter([x-.03,x,x+.03],values,color=color,s=20,alpha=.55)
            axes[0].plot([x-.07,x+.07],[statistics.median(values)]*2,color=color,linewidth=3,label=name if i==0 else None)
            data.append(dict(figure='factorial',label=label,contrast=key,paired_speedups=values))
        for j,(key,name,color) in enumerate((('batch_current_stream__rope_upstream','Batch','#6b7280'),
                ('async_priority_current_stream__rope_upstream','Streaming','#0284c7'))):
            vals=lane['arms'][key]['first_packet_muxed_s'];x=i+(j-.5)*.28
            axes[1].scatter([x-.025,x,x+.025],vals,color=color,s=22,alpha=.55)
            axes[1].plot([x-.08,x+.08],[statistics.median(vals)]*2,color=color,linewidth=3,label=name if i==0 else None)
            data.append(dict(figure='first_packet',label=label,arm=key,seconds=vals))
    for axis in axes:
        axis.set_xticks(range(4),labels,rotation=20,ha='right');axis.grid(axis='y',alpha=.2);axis.legend(fontsize=8)
    axes[0].axhline(1,color='#dc2626',linestyle='--',linewidth=1)
    axes[0].set_ylabel('Paired complete-service speedup');axes[0].set_title('Three blocked repeats; negative samples retained')
    axes[1].set_yscale('log');axes[1].set_ylabel('First MP4 packet muxed (s, log)');axes[1].set_title('Same original RoPE; not client display latency')
    save(fig,'h200_complete_service_factorial','48条实际H200视频；同一process内分块随机顺序重复，不是独立机器重复。计时含生成、VAE及增量编码flush，不含load/事后审计。全部配对noise/routes/latent/RGB完全一致。散点为3次观测，横线为中位数；首包不是用户屏幕显示时间。')

    fig,axes=plt.subplots(1,2,figsize=(10.5,4));mode_names=['serial','same_thread','producer']
    for i,(queries,name) in enumerate(((4680,'Page256 / cap4 / high reuse'),(1560,'Page64 / cap2 / medium reuse'))):
        eager=load(prefix+f'page_producer_q{queries}_v1/summary.json')
        graph=load(prefix+f'page_graph_q{queries}_v1/summary.json')
        modes=list(graph['samples'])
        for j,(record,label,color) in enumerate(((eager,'Eager consumer','#6b7280'),(graph,'CUDA-graph consumer','#0284c7'))):
            vals=[1000*statistics.median(x['full_wall_s'] for x in record['samples'][mode]) for mode in modes]
            axes[i].plot(range(len(modes)),vals,'o-',label=label,color=color)
            data.append(dict(figure='bounded_pages',queries=queries,consumer=label,modes=modes,median_ms=vals))
        axes[i].set_xticks(range(len(modes)),modes,rotation=15);axes[i].set_title(f'Q{queries}: '+name);axes[i].grid(alpha=.2);axes[i].legend(fontsize=8)
    axes[0].set_ylabel('Pack + H2D + partial Attention wall (ms)')
    save(fig,'bounded_page_pipeline','每点5次warmup后30次测量，compile/capture单列；包含frame-major CPU pack、有限pinned/GPU槽及partial Attention，不含RoPE/在线route。图展示dispatch组织和生产者的独立影响，不是与最佳eager-union或完整视频的胜负。')

    observer=load('videos/sprint24h_20260907/recache_version_capture39_v1/summary.json')
    event=next(r for r in observer['variants'] if r['variant']=='official_recache')['recache_events'][-1]
    fig,axis=plt.subplots(figsize=(6,3.8));layers=[r['layer'] for r in event['representation_versions']]
    for key,label in [('K_error','Key'),('V_error','Value')]:
        values=[r[key]['relative_l2'] for r in event['representation_versions']]
        axis.plot(layers,values,'o-',label=label)
        data.append(dict(figure='KV_versions',at_latent=event['at_latent'],layers=layers,kind=label,relative_l2=values))
    axis.set_xlabel('Transformer layer');axis.set_xticks(layers);axis.set_ylabel('Pre/post recache relative L2')
    axis.set_title('Same completed visual frames, different KV versions');axis.grid(alpha=.2);axis.legend()
    save(fig,'same_frame_KV_versions','官方recache前后同一组已完成帧；观测器对整个noise/latent/RGB轨迹无扰动。条件与重算上下文共同变化，不能归因为纯文本效应；此图也不证明召回哪种版本质量更好。')
    report=dict(status='pass',sources=sources,data=data,figures=figures,scope='development_evidence_not_new_algorithm_promotion')
    (args.output/'figure_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    sections=''.join('<section><h2>'+html.escape(f['name'].replace('_',' '))+'</h2><img src="'+f['name']+'.svg"><p>'+html.escape(f['caption'])+'</p></section>' for f in figures)
    (args.output/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>LongLive系统与记忆证据</title><style>body{font:16px/1.65 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#243043}img{width:100%;height:auto}section{margin:42px 0}p{color:#4b5563}</style><h1>系统收益与记忆版本：开发集证据</h1><p>正结果、负结果和证据边界并列；目标仍是更快、更好的因果流式长视频。</p>'+sections+'<p><a href="figure_manifest.json">完整数据和来源SHA</a></p></html>')
    print(json.dumps(dict(status='pass',figures=len(figures),output=str(args.output))))


if __name__=='__main__':main()
