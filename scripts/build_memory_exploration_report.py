#!/usr/bin/env python3
"""Build evidence figures and a terminal audit for the completed exploration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def read(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    results=Path(args.results)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    probe=results/'metrics/memory_dynamics_d6b20e4'
    analysis=results/'metrics/memory_dynamics_analysis_79b83a3'
    followup=results/'metrics/proxy_order_followup_cc8f2e9'
    phase=results/'metrics/clean_phase_proxy_79b83a3'
    gates=results/'metrics/aligned_final_gate_9e16f23'
    expected=[]
    for name in ('gate_motion39','motion120','state120'):
        expected.append((f'generator_{name}',probe/name/'terminal.json'))
    for kind in ('motion','state'):
        expected.append((f'render_{kind}',analysis/kind/'visuals/render_audit.json'))
        for layer in (0,19):
            expected.append((f'permutation_{kind}_{layer}',followup/kind/f'permutation_layer{layer:02d}.json'))
            for start in (46800,177840):
                expected.extend([(f'shadow_{kind}_{layer}_{start}',analysis/kind/f'layer{layer}_start{start}.json'),
                    (f'factorial_{kind}_{layer}_{start}',followup/kind/f'factorial_layer{layer}_start{start}.json'),
                    (f'phase_{kind}_{layer}_{start}',phase/kind/f'layer{layer:02d}_start{start:08d}_clean.json')])
    for size in ('small','large'):
        expected.append((f'gate_{size}',gates/f'{size}.json'))
    audit=[]
    for name,path in expected:
        if not path.exists():
            audit.append({'id':name,'status':'missing','path':str(path)})
        else:
            data=read(path)
            audit.append({'id':name,'status':data['status'],'path':str(path.resolve()),
                          'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    terminal={'scope':'completed_exploration_artifacts_not_entire_LongLive_plan',
        'expected':len(expected),'pass':sum(r['status']=='pass' for r in audit),
        'fail':sum(r['status']=='fail' for r in audit),'missing':sum(r['status']=='missing' for r in audit),
        'artifacts':audit}
    (out/'terminal_audit.json').write_text(json.dumps(terminal,indent=2)+'\n')
    if terminal['pass']!=terminal['expected']:
        raise RuntimeError('incomplete exploration artifacts; inspect terminal audit')
    variants=['raw_first','raw_current','aligned_first','aligned_current']
    rows=[]
    for kind in ('motion','state'):
        for layer in (0,19):
            for start in (46800,177840):
                data=read(followup/kind/f'factorial_layer{layer}_start{start}.json')
                rows.append({'prompt':kind,'layer':layer,'latent':start//1560,
                             **{k:data['records'][k]['output_error']['relative_l2'] for k in variants}})
    with (out/'proxy_refresh_factorial.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    ratios=np.array([[r[k]/r['raw_first'] for k in variants] for r in rows])
    fig,ax=plt.subplots(figsize=(8.5,5.6))
    im=ax.imshow(ratios,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=.8,vcenter=1,vmax=1.7),aspect='auto')
    ax.set_xticks(range(4),['Raw\nfirst Q','Raw\ncurrent Q','Aligned\nfirst Q','Aligned\ncurrent Q'])
    ax.set_yticks(range(8),[f"{r['prompt']} L{r['layer']} latent{r['latent']}" for r in rows])
    for y in range(8):
        for x in range(4):
            ax.text(x,y,f'{ratios[y,x]:.3f}x',ha='center',va='center',fontsize=10)
    ax.set_title('Clean-context error: representation matters more than refresh')
    fig.colorbar(im,ax=ax,label='FP32 output relative-L2 / raw-first (lower is better)')
    fig.text(.02,.01,'Retrieved-context teacher; 8 correlated layer/chunk points from 2 development trajectories. No video-quality claim.',fontsize=8)
    fig.tight_layout(rect=(0,.03,1,1))
    fig.savefig(out/'proxy_refresh_factorial.png',dpi=180)
    fig.savefig(out/'proxy_refresh_factorial.pdf')
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    summaries={kind:read(probe/f'{kind}120_summary.json') for kind in ('motion','state')}
    for kind,marker in (('motion','o'),('state','x')):
        curve=summaries[kind]['archive_growth']['rows']
        axes[0].plot([r['committed_latents'] for r in curve],[r['kv_bytes']/2**30 for r in curve],
                     label=kind,marker=marker,markevery=5,markersize=4)
        observations=read(probe/f'{kind}120/route_observations/observations.json')['records']
        for layer,style in ((0,'-'),(19,'--')):
            medians=[statistics.median(r['executed_vs_fresh_tokens']['jaccard'] for r in observations
                                      if r['layer']==layer and r['denoising_pass']==call and r['fresh_shadow_computed'])
                     for call in (1,2,3,4)]
            axes[1].plot(range(1,5),medians,style,marker=marker,label=f'{kind}, L{layer}')
    axes[0].set(xlabel='Committed latent frames',ylabel='Actual CPU KV archive (GiB)',
                title='Bounded pinned staging is not bounded archive')
    axes[1].set(xlabel='Call after first route selection',ylabel='Token-coordinate Jaccard vs first route',
                title='Natural raw-proxy routes are not cache-enforced routes',ylim=(0,1))
    axes[1].set_xticks([1,2,3,4],['denoise1','denoise2','denoise3','clean commit'])
    for ax in axes:
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'archive_and_denoising.png',dpi=180);fig.savefig(out/'archive_and_denoising.pdf');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,kind in zip(axes,('motion','state')):
        data=read(probe/f'{kind}120/terminal.json')
        baseline_set=set(data['pulses'][0]['retrievals'][0]['pool_indices'][0])
        for pulse in data['pulses'][1:]:
            same=baseline_set==set(pulse['retrievals'][0]['pool_indices'][0])
            label=pulse['policy']+(' (same set, reordered)' if same else '')
            ax.plot(range(3),[r['relative_l2'] for r in pulse['per_chunk']],marker='o',label=label)
        ax.set(title=kind,ylabel='Latent relative-L2 vs no-intervention trajectory')
        ax.set_xticks(range(3),['pulse chunk','ordinary +1','ordinary +2'])
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('A one-chunk retrieval pulse can change later trajectories; divergence is not quality')
    fig.tight_layout();fig.savefig(out/'pulse_response.png',dpi=180);fig.savefig(out/'pulse_response.pdf');plt.close(fig)
    improvement=[1-r['aligned_first']/r['raw_first'] for r in rows]
    summary={'status':'pass','terminal':{k:v for k,v in terminal.items() if k!='artifacts'},
        'aligned_first_clean_capture_improvement':{'points':8,'positive':sum(x>0 for x in improvement),
            'min':min(improvement),'median':statistics.median(improvement),'max':max(improvement)},
        'prototype_refresh_video_validation':'separate_cf5c25f_local_six_case_batch',
        'formal_holdouts_used':False,'entire_plan_complete':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
