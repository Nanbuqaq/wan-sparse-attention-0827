#!/usr/bin/env python3
"""Source-hashed development evidence figures; no automatic paper winner."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--metrics-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    sources=[];figures=[];data=[]
    def load(relative):
        path=args.metrics_root/relative;raw=path.read_bytes()
        sources.append(dict(path=str(path.resolve()),sha256=hashlib.sha256(raw).hexdigest()))
        return json.loads(raw)
    def save(fig,name,caption):
        fig.savefig(args.output/(name+'.svg'),bbox_inches='tight')
        fig.savefig(args.output/(name+'.png'),bbox_inches='tight',dpi=160)
        plt.close(fig);figures.append(dict(name=name,caption=caption))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.dpi':110,'svg.fonttype':'none'})
    layers=[0,9,19,29]
    routes=[('legacy_final25','Final25'),('spatial_quadrants_shared_0.25','Spatial shared25'),
            ('spatial_quadrants_per_group_0.25','Spatial conditional25'),('query_features_per_group_0.25','Q-feature conditional25')]
    fig,axes=plt.subplots(2,2,figsize=(10,6),sharex=True)
    for col,kind in enumerate(('motion','state')):
        captures=[load(f'group_relations_{kind}'+('_middle_v1' if layer in (9,19) else '_v1')+f'/layer{layer:02d}.json') for layer in layers]
        for key,label in routes:
            errors=[c['rows'][key]['output_error']['relative_l2'] for c in captures]
            union=[100*c['rows'][key]['unique_transfer_density'] for c in captures]
            axes[0,col].plot(layers,errors,'o-',label=label)
            axes[1,col].plot(layers,union,'o-',label=label)
            data.append(dict(figure='grouping',kind=kind,route=key,layers=layers,relative_l2=errors,union_percent=union))
        axes[0,col].set_title(kind.capitalize()+' complete captures')
        axes[1,col].set_xlabel('Transformer layer');axes[1,col].set_xticks(layers)
        axes[0,col].grid(alpha=.2);axes[1,col].grid(alpha=.2)
    axes[0,0].set_ylabel('Attention output relative L2');axes[1,0].set_ylabel('Unique history KV union (%)')
    axes[0,1].legend(fontsize=8)
    fig.suptitle('Conditional consumption trades additional physical KV for approximation',fontsize=13)
    fig.tight_layout()
    save(fig,'grouping_error_and_union','两类别完整capture包含exact/current/recent。各方法历史pair预算均25%，但conditional的物理union更大；不是等字节质量比较，更不是完整视频质量证明。')

    fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
    for axis,kind in zip(axes,('motion','state')):
        captures=[load(f'feature_prototypes_{kind}_v1/layer{layer:02d}.json') for layer in layers]
        for key,label in [('legacy25_random_groups','Random groups'),('legacy25_spatial_groups','Spatial groups'),('legacy25_key_kmeans','K-feature groups')]:
            values=[c['results'][key]['error']['relative_l2'] for c in captures]
            axis.plot(layers,values,'o-',label=label)
            data.append(dict(figure='representation',kind=kind,variant=key,layers=layers,relative_l2=values))
        axis.set_title(kind.capitalize());axis.set_xlabel('Transformer layer');axis.set_xticks(layers);axis.grid(alpha=.2)
    axes[0].set_ylabel('Attention output relative L2');axes[1].legend()
    fig.suptitle('Equal prototype slots: information grouping improves representation',fontsize=13)
    fig.tight_layout()
    save(fig,'equal_slot_representation','同一legacy raw25%选择，Block64内均4个原型槽位。分组只用已提交K；K特征分组在8个capture均优于空间/随机分组。此图不包含实时索引成本，也不证明长期视频收益。')

    budget=load('group_budget477_review_v1/summary.json')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for axis,kind in zip(axes,('motion','state')):
        group=next(g for g in budget['groups'] if g['seed']==20260908 and g['prompt']=='calibration_'+kind)
        rows={r['variant']:r for r in group['cases']}
        reference=rows['Dense']['actual_history_H2D_bytes']
        for name in ('Dense','Final25','Final50','shared50','per_group25'):
            row=rows[name];x=100*row['actual_history_H2D_bytes']/reference;y=row['complete_wall_s']
            offsets={'per_group25':(-65,16),'Final50':(10,1),'shared50':(10,-12)}
            label='Conditional25' if name=='per_group25' else name
            axis.scatter([x],[y],s=55);axis.annotate(label,(x,y),xytext=offsets.get(name,(5,5)),textcoords='offset points',fontsize=8)
            data.append(dict(figure='video_budget',kind=kind,variant=name,physical_history_percent=x,wall_s=y))
        axis.set_xlim(15,112);axis.set_ylim(165,216);axis.set_title(kind.capitalize()+' / development seed20260908')
        axis.set_xlabel('History H2D bytes / Dense (%)');axis.grid(alpha=.2)
    axes[0].set_ylabel('Complete 477-frame workflow (s)')
    fig.suptitle('Video control: conditional25 is not a demonstrated system winner',fontsize=13)
    fig.tight_layout()
    save(fig,'new_seed_video_budget','同类别各臂在同一RTX4090 lane运行，n=1，完整耗时含结果保存/审计，不是独立重复计时。conditional25实际搬运43–47%Dense历史KV；未显示稳定质量优势。旧seed混合硬件控制未画入。')

    report=dict(status='pass',scope='development_characterization_not_formal_promotion',sources=sources,data=data,figures=figures)
    (args.output/'figure_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    sections=''.join(f'<section><h2>{html.escape(f["name"].replace("_"," "))}</h2><img src="{f["name"]}.svg"><p>{html.escape(f["caption"])}</p></section>' for f in figures)
    document='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>LongLive探索证据</title><style>body{font:16px/1.65 system-ui;max-width:1080px;margin:40px auto;padding:0 24px;color:#243043}img{width:100%;height:auto}section{margin:40px 0}p{color:#4b5563}h1{font-size:28px}</style><h1>分组、表示与完整执行：开发集证据</h1><p>目标是更快、更好的流式长视频。支持与负面证据并列；当前没有宣布新算法成为长期质量赢家。</p>'+sections+'<p>全部数据与来源SHA：<a href="figure_manifest.json">figure_manifest.json</a>。方法名遮挡的首次视觉审查保留在原始审查目录。</p></html>'
    (args.output/'index.html').write_text(document)
    print(json.dumps(dict(status='pass',figures=len(figures),sources=len(sources),output=str(args.output))))


if __name__=='__main__':main()
