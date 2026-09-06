#!/usr/bin/env python3
"""Close explicit development gaps without selecting from completed holdouts."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();rows=[];prefetch={};feedback=[];sources=[]
    for kind in ('motion','state'):
        path=args.root/kind/'summary.json';data=json.loads(path.read_text())
        if data['status']!='pass' or data['missing'] or len(data['records'])!=6:
            raise ValueError('all twelve complete capture evaluations required')
        grouped=defaultdict(list)
        for capture in data['records']:
            records=capture['records']
            if len(records)!=13 or not capture['all_routes_built_before_teacher']:
                raise ValueError('missing ablation or invalid teacher boundary')
            for name,value in records.items():
                if name=='legacy_cap25' or name.startswith('matched_legacy__'):continue
                control=records['matched_legacy__'+name]
                for key in ('tokens_per_head','payload_bytes','rectangular_bytes','scheduled_pairs'):
                    if value[key]!=control[key]:raise ValueError('unfair realized-budget comparison')
                grouped[name].append({'error_ratio':value['output_error']['relative_l2']/control['output_error']['relative_l2'],
                    'density':value['logical_density'],'payload':value['payload_bytes'],
                    'retained_total_mass':value['probability_mass']['retained_total_mass_mean']})
        for name,items in grouped.items():
            ratios=[r['error_ratio'] for r in items]
            rows.append({'category':kind,'candidate':name,'captures':len(items),
                'mean_relative_l2_ratio_vs_byte_matched_legacy':statistics.mean(ratios),
                'worst_relative_l2_ratio_vs_byte_matched_legacy':max(ratios),
                'nonregressing_calls':sum(r<=1 for r in ratios),
                'realized_density_min':min(r['density'] for r in items),'realized_density_max':max(r['density'] for r in items),
                'payload_min':min(r['payload'] for r in items),'payload_max':max(r['payload'] for r in items)})
        prefetch[kind]=data['prefetch_summary']
        feedback += [{'category':kind,**r} for r in data['censored_feedback_gate']]
        sources.append({'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    names={r['candidate'] for r in rows}
    eligible=[name for name in sorted(names) if all(r['worst_relative_l2_ratio_vs_byte_matched_legacy']<=1 for r in rows if r['candidate']==name)]
    result={'status':'pass','full_capture_evaluations':12,'ablation_comparisons':72,'missing':0,
        'budget_ablations':rows,'prefetch':prefetch,'censored_feedback':feedback,'sources':sources,
        'new_route_promoted':False,'formal_configs_changed':False,
        'q_to_next_proto_decision':'screen-only direct cross-layer proxy; measured recall/traffic are above; no timed/video prefetch promoted',
        'both_category_worst_case_nonregressing_variants':eligible,
        'coverage_decision':'no variant passes both categories worst-case byte-matched nonregression; do not tune completed holdouts' if not eligible else 'local candidates exist but have no on-policy video promotion',
        'exploration_definition':'5% existing deterministic age-prior allocation, not uniform random exploration or UCB',
        'feedback_decision':'some observed reentry lower bounds exceed10%; censored observations merit future controlled study, not a safe-eviction label or immediate bandit',
        'scope':'two fixed development trajectories; correlated layer/chunk observations; no on-policy video or speed claim'}
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    for ax,kind in zip(axes,('motion','state')):
        group=sorted([r for r in rows if r['category']==kind],key=lambda r:r['candidate'])
        x=np.arange(len(group))
        ax.plot(x,[r['mean_relative_l2_ratio_vs_byte_matched_legacy'] for r in group],marker='o',label='mean')
        ax.plot(x,[r['worst_relative_l2_ratio_vs_byte_matched_legacy'] for r in group],marker='D',label='worst')
        ax.axhline(1,color='gray',linestyle='--')
        ax.set_xticks(x,[r['candidate'].replace('__','\n').replace('count_uniform','count').replace('peak_value','peak') for r in group],rotation=30,fontsize=8)
        ax.set(title=kind,ylabel='FP32 output relative-L2 / exact-byte-matched legacy (lower better)')
        ax.legend()
    fig.savefig(args.output/'budget_ablations.png',dpi=150);plt.close(fig)
    pm,ps=(prefetch[k]['q_to_next_proto'] for k in ('motion','state'))
    feedback_text=', '.join(f"{r['category']} L{r['layer']}: {r['observed_lower_bound_reentry_fraction']:.2%}" for r in feedback)
    text=['# Remaining contract experiments: audited outcomes','','12 complete captures, 72 utility/control comparisons; all controls match actual token counts, payload, padding and scheduled logical pairs.',
        'No formal prompts or parameters were changed. These post-formal gap-closure diagnostics are not retrospectively part of the earlier method-selection protocol.','',
        f"- Direct current-layer Q to next-layer raw K-prototype scores: byte recall{pm['byte_recall']:.2%}/{ps['byte_recall']:.2%}, traffic{pm['traffic_ratio_vs_exact']:.3f}x/{ps['traffic_ratio_vs_exact']:.3f}x of exact no-prefetch. No target Q, target route or future prototype entered prediction.",
        '- Source and target head spaces are not guaranteed aligned. Negative results concern this direct proxy, not every cross-layer predictor. Timeliness/stall are not fabricated for the unexecuted prefetch; high-extra-traffic candidates were not advanced to video.',
        '- Both-category worst-case nonregressing candidates: '+(', '.join(eligible) if eligible else 'none')+'. Mean error is reported separately from the worst-case criterion.',
        '- Observed next-two-chunk teacher-top reentry lower bounds: '+feedback_text+'. Coarse omission leaves many blocks unobserved; conditioning only on observable blocks gives higher rates and a different denominator.',
        '- This establishes a censored-feedback concern, not a learned policy, causal semantic importance, or a safe rule for deleting the CPU archive.','',
        '![Budget ablations](budget_ablations.png)','']
    (args.output/'REPORT.md').write_text('\n'.join(text))
    print(json.dumps({k:v for k,v in result.items() if k not in ('budget_ablations','prefetch','censored_feedback','sources')},indent=2))


if __name__=='__main__':main()
