#!/usr/bin/env python3
"""Figures and terminal audit for matched inputs and causal bootstrap evidence."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    root=Path(args.results);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    replay=root/'metrics/cross_trajectory_replay_f7d81b4_local'
    cases=[];rows=[]
    for lane in range(2):
        terminal=json.loads((replay/f'lane{lane}/terminal.json').read_text())
        if terminal['status']!='pass' or terminal['missing']!=0:raise ValueError('incomplete replay lane')
        for state in terminal['cases']:
            data=json.loads(Path(state['output']).read_text())
            ratios=[]
            for row in data['rows']:
                old=row['methods']['transfer_vaware_hybrid_history']['output_error']['relative_l2']
                new=row['methods']['rope_aligned_final_history']['output_error']['relative_l2']
                ratios.append(new/old)
                rows.append({'actor':data['actor'],'prompt':data['prompt'],'layer':data['layer'],
                    'latent':data['current_start']//1560,'call':row['call'],'raw_error':old,'aligned_error':new,'ratio':new/old})
            cases.append({'actor':data['actor'],'prompt':data['prompt'],'layer':data['layer'],
                          'latent':data['current_start']//1560,'mean_ratio':statistics.mean(ratios),
                          'worse_calls':sum(r>1.0001 for r in ratios)})
    if len(cases)!=48 or len(rows)!=240:raise ValueError('expected48 sequences240 calls')
    with (out/'cross_trajectory.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    actors=['rag_dense','transfer_vaware_hybrid_history','rope_aligned_final_history']
    fig,axes=plt.subplots(1,2,figsize=(12,5.2),sharey=True)
    for ax,kind in zip(axes,['motion','state']):
        matrix=[]
        for actor in actors:
            for latent in [18,30]:
                matrix.append([next(c['mean_ratio'] for c in cases if c['actor']==actor and c['prompt']==f'calibration_{kind}'
                                    and c['latent']==latent and c['layer']==layer) for layer in [0,9,19,29]])
        matrix=np.array(matrix)
        im=ax.imshow(matrix,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=.88,vcenter=1.,vmax=1.04),aspect='auto')
        for y in range(6):
            for x in range(4):ax.text(x,y,f'{matrix[y,x]:.3f}',ha='center',va='center')
        ax.set_xticks(range(4),['L0','L9','L19','L29'])
        ax.set_yticks(range(6),[f'{actor}: {stage}' for actor in ['Dense path','Final path','Aligned path']
                               for stage in ['1-frame startup','6-frame history']])
        ax.set_title(kind)
    fig.suptitle('Aligned / raw local error on exactly matched39-latent trajectories (lower is better)')
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,label='Mean over five calls; correlated observations')
    fig.subplots_adjust(left=.21,right=.82,wspace=.12,bottom=.1,top=.88)
    fig.savefig(out/'matched_cross_trajectory.png',dpi=180);fig.savefig(out/'matched_cross_trajectory.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    comparison=[]
    for ax,kind in zip(axes,['motion','state']):
        base=json.loads((root/f'metrics/aligned_final_quality_cf5c25f/{kind}.json').read_text())['rows']
        ablations=json.loads((root/f'metrics/bootstrap_quality_2ad4465/{kind}.json').read_text())['rows']
        values=[next(r['lpips_mean'] for r in base if r['method']=='transfer_vaware_hybrid_history'),
                next(r['lpips_mean'] for r in base if r['method']=='rope_aligned_final_history'),
                next(r['lpips_mean'] for r in ablations if r['method_params']['bootstrap_layer']==9),
                next(r['lpips_mean'] for r in ablations if r['method_params']['bootstrap_layer']==-1)]
        ax.bar(range(4),values,color=['gray','#65a9d8','#f3b56a','#6bb88b'])
        ax.set_xticks(range(4),['Final','Aligned','L9 raw\nstartup','All raw\nstartup']);ax.set(title=kind,ylabel='LPIPS to same-GPU Dense')
        for i,value in enumerate(values):ax.text(i,value+.001,f'{value:.4f}',ha='center',fontsize=9)
        ax.set_ylim(0,max(values)*1.2)
        comparison.append({'prompt':kind,'values':values})
    fig.suptitle('Causal bootstrap intervention: original development seed only, not formal promotion')
    fig.tight_layout();fig.savefig(out/'bootstrap_causal_quality.png',dpi=180);fig.savefig(out/'bootstrap_causal_quality.pdf');plt.close(fig)
    trajectory=json.loads((root/'metrics/matched_trajectory_capture_be00491/trajectory_audit.json').read_text())
    causal=json.loads((root/'videos/bootstrap_ablation_2ad4465_local/causal_audit.json').read_text())
    result={'status':'pass','original_trajectories':len(trajectory['cases']),'trajectory_gate':trajectory['status'],
        'replay_sequences':len(cases),'replay_calls':len(rows),'bootstrap_causal_gate':causal['status'],
        'bootstrap_videos':len(causal['cases']),'cross_context_summary':cases,'quality':comparison,
        'formal_promotion':False,'long_validation_still_required':True}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cross_context_summary','quality')},indent=2))


if __name__=='__main__':main()
