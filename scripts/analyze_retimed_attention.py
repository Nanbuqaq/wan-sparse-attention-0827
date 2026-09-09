#!/usr/bin/env python3
"""Compare actual position-rebound Attention to an already verified capture.

Frozen old inputs are only a causal witness at first-denoise L0. Later actual
queries and current K/V can change through the generation trajectory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.native_temporal_rephase import rephase_temporal_keys
from scripts.analyze_native_attention_teacher import decompose


def input_witness(original,retimed,delta):
    width=8*original['frame_tokens'];source=slice(width,2*width)
    if original['k'].shape!=retimed['k'].shape or original['k'].shape[1]!=4*width:
        raise ValueError('expected same initial/source/away/current partition')
    other=torch.cat((original['k'][:,:width],original['k'][:,2*width:]),dim=1)
    new_other=torch.cat((retimed['k'][:,:width],retimed['k'][:,2*width:]),dim=1)
    return dict(Q_bitwise_equal=torch.equal(original['q'],retimed['q']),
        source_V_bitwise_equal=torch.equal(original['v'][:,source],retimed['v'][:,source]),
        all_V_bitwise_equal=torch.equal(original['v'],retimed['v']),
        all_other_K_bitwise_equal=torch.equal(other,new_other),
        source_K_equals_declared_temporal_rotation=torch.equal(rephase_temporal_keys(original['k'][:,source],delta),retimed['k'][:,source]),
        source_spatial_K_bitwise_equal=torch.equal(original['k'][:,source,...,44:],retimed['k'][:,source,...,44:]))


def load_verified(root):
    path=root/'summary.json';report=json.loads(path.read_text())
    if report['status']!='pass' or not report.get('observer_noise_latent_RGB_equivalence'):
        raise ValueError('capture must match full generated latent and RGB before analysis')
    payload=root/'attention_teacher.pt'
    with payload.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
    if digest!=report['attention_teacher']['sha256']:raise ValueError('capture SHA changed')
    data=torch.load(payload,map_location='cpu',weights_only=True)
    assert data['schema']=='native_attention_teacher_v1' and data['online_routing_may_not_access']
    records={(r['phase'],r['layer']):r for r in data['records']}
    assert len(records)==len(data['records'])==9
    return report,records,digest,hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--retimed',type=Path,required=True)
    p.add_argument('--original-analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.set_num_interop_threads(1)
    original,old,old_sha,old_summary_sha=load_verified(args.original)
    retimed,new,new_sha,new_summary_sha=load_verified(args.retimed)
    for key in ('seed','noise_sha256','pre_return_latent_sha256','prompts_per_block','source_files_sha256','triton_version','fixed_native_adaln_recipe'):
        assert original[key]==retimed[key],key
    assert new.keys()==old.keys() and retimed['episode_memory']['installation']['admission_plan']['temporal_delta']==64
    analysis_path=args.original_analysis/'role_diagnostics.json';old_analysis=json.loads(analysis_path.read_text())
    previous=next(c for c in old_analysis['cases'] if c['arm']=='shot')
    assert previous['capture_sha256']==old_sha and previous['all_FP32_replay_gates']
    previous_rows={(r['phase'],r['layer']):r for r in previous['rows']}
    rows=[]
    for key in sorted(new):
        witness=input_witness(old[key],new[key],64)
        assert witness['source_V_bitwise_equal'] and witness['source_K_equals_declared_temporal_rotation'] and witness['source_spatial_K_bitwise_equal']
        # Only this anchor has not yet propagated the intervention through layers or denoising.
        if key==(0,0):assert all(witness.values()),witness
        report,_=decompose(new[key],['initial','source','away','current'])
        assert report['FP32_replay_gate'],report['FP32_replay_error']
        previous_row=previous_rows[key]
        rows.append(dict(phase=key[0],layer=key[1],input_witness=witness,
            original_roles=previous_row['roles'],retimed_roles=report['roles'],
            retimed_FP32_error=report['FP32_replay_error'],retimed_FP32_gate=report['FP32_replay_gate']))
        print(json.dumps(dict(phase=key[0],layer=key[1],source_mass_original=previous_row['roles']['source']['probability_mass']['mean'],
            source_mass_retimed=report['roles']['source']['probability_mass']['mean'])),flush=True)
    result=dict(status='pass',records=9,rows=rows,first_denoise_L0_only_source_temporal_K_changed=True,
        sources=dict(original_capture_sha256=old_sha,retimed_capture_sha256=new_sha,original_summary_sha256=old_summary_sha,
            retimed_summary_sha256=new_summary_sha,original_analysis_sha256=hashlib.sha256(analysis_path.read_bytes()).hexdigest()),
        query_sample='32 fixed geometric queries per head, all24heads; not full Q average',
        timing_comparison_allowed=False,semantic_importance_not_proven_by_mass=True,
        late_queries_and_current_KV_can_change=True,clean_commit_is_not_a_fifth_denoising_step=True)
    (args.output/'actual_attention_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors={'initial':'#9d98a1','source':'#d94c54','away':'#63a56d','current':'#4686c3'}
    fig,axes=plt.subplots(2,3,figsize=(12,6),sharey=True)
    for row_index,(field,label) in enumerate((('original_roles','Original position'),('retimed_roles','Recent binding'))):
        for col,phase in enumerate((0,3,4)):
            selected=sorted((r for r in rows if r['phase']==phase),key=lambda r:r['layer'])
            bottom=np.zeros(3);ax=axes[row_index,col]
            for role,color in colors.items():
                values=np.array([r[field][role]['probability_mass']['mean']*100 for r in selected])
                ax.bar(range(3),values,bottom=bottom,color=color,label=role);bottom+=values
            ax.set_xticks(range(3),['L0','L14','L29']);ax.set_ylim(0,100)
            ax.set_title(label+' | '+{0:'first denoise',3:'last denoise',4:'clean commit'}[phase])
            if col==0:ax.set_ylabel('Sampled attention mass (%)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(args.output/'actual_role_mass.png',dpi=150);plt.close(fig)
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>重绑定实际Attention</title><style>body{font:17px system-ui;max-width:1200px;margin:30px auto}img{max-width:100%}</style><h1>位置改变后，实际Attention如何变化？</h1><p>每头32个固定几何query、全部24heads和真实K/V。两capture均整段输出等价；first-denoise L0验证仅source时间K变化，其他输入逐位相同。后续层与去噪阶段的Q及current KV会沿轨迹改变，不能当固定Q实验。</p><img src="actual_role_mass.png"><p>概率质量不是语义重要性，clean commit也不是第五次去噪。计入capture的耗时不能用作方法速度对比。</p><a href="actual_attention_comparison.json">数值门禁、输入见证、全部角色统计与SHA</a>')


if __name__=='__main__':main()
