#!/usr/bin/env python3
"""Small source-linked figure for the actual native Attention role diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from scripts.analyze_native_attention_teacher import output_error
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    source=args.analysis/'role_diagnostics.json';d=json.loads(source.read_text());errors=[]
    colors={'initial':'#9d98a1','source':'#d94c54','away':'#63a56d','current':'#4686c3'}
    fig,axes=plt.subplots(2,3,figsize=(12,6),sharey=True)
    lines=['# 初始、源、离开场景与当前chunk的竞争','',
           '两次capture全部latent/RGB与原控制逐位一致。32个固定几何query/head、全部24heads、',
           '全部实际K/V；layers0/14/29，first/last denoising与clean commit。不是全Q均值，也不是语义mask。','']
    for row,case in enumerate(d['cases']):
        derived=torch.load(args.analysis/f'{case["arm"]}__group_decomposition.pt',map_location='cpu',weights_only=True)
        for r,t in zip(case['rows'],derived):
            errors.append(dict(arm=case['arm'],phase=r['phase'],layer=r['layer'],
                               error=output_error(t['fp32_output'],t['native_output'])))
        for col,phase in enumerate((0,3,4)):
            entries=sorted((r for r in case['rows'] if r['phase']==phase),key=lambda r:r['layer'])
            bottom=np.zeros(3);ax=axes[row,col]
            for role,color in colors.items():
                values=np.array([r['roles'].get(role,{}).get('probability_mass',{}).get('mean',0)*100 for r in entries])
                ax.bar(range(3),values,bottom=bottom,color=color,label=role);bottom+=values
            ax.set_xticks(range(3),['L0','L14','L29']);ax.set_ylim(0,100)
            ax.set_title(f'{case["arm"]} | '+{0:'first denoise',3:'last denoise',4:'clean commit'}[phase])
            if col==0:ax.set_ylabel('Sampled attention mass (%)')
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='upper center')
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(args.output/'attention_role_mass.png',dpi=160);plt.close(fig)
    worst={k:max(e['error'][k] for e in errors) for k in ('max_abs','relative_l2','one_minus_cosine')}
    result=dict(source_analysis_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),records=18,
                scalar_distance_reductions='FP64; Attention computation remains FP32',worst_native_output_error=worst,rows=errors,
                semantic_causality_not_proven=True)
    (args.output/'numeric_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    lines+=['| reset / last denoise | initial | retrieved source | current |','|---|---:|---:|---:|']
    reset=next(c for c in d['cases'] if c['arm']=='reset_reveal')
    for r in reset['rows']:
        if r['phase']==3:
            m={k:v['probability_mass']['mean']*100 for k,v in r['roles'].items()}
            lines.append(f'| layer{r["layer"]} | {m["initial"]:.1f}% | {m["source"]:.1f}% | {m["current"]:.1f}% |')
    lines+=['','18条FP32回放均通过BF16原算子输出门禁。标量cosine/norm用FP64重新归约，',
            '避免v1 FP32归约出现约1e-6的负cosine距离；原v1分析文件不改写。',
            '', '可支持的问题：移除away后，初始上下文仍可能与源状态竞争；统计也明显依赖denoising阶段。',
            '不能据此直接证明空罐锚点导致丢失，也不能把source占比当质量或在线utility。',
            '下一步是明示初始锚点退出的受控视频，以及必要时对source内的状态/背景区域分解。','']
    (args.output/'INTERPRETATION.md').write_text('\n'.join(lines))
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Attention角色诊断</title><style>body{font:17px system-ui;max-width:1200px;margin:30px auto;padding:0 20px}img{width:100%}</style><h1>信息已取回，仍未被稳定使用</h1><p>18组实际Q/K/V/O通过FP32回放；仅固定几何query采样，不是全Q均值。两次capture整段输出与原控制逐位相同。</p><img src="attention_role_mass.png"><p>末次去噪的layer14：initial约25.1%、source约11.8%；layer29约96.9%来自current。它是竞争/阶段依赖的证据，不是语义因果证明。</p><a href="INTERPRETATION.md">解释与边界</a> · <a href="numeric_audit.json">数值审计</a>')
    print(json.dumps(worst))


if __name__=='__main__':main()
