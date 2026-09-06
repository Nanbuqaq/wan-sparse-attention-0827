#!/usr/bin/env python3
"""Reproducible figures joining same-route timings and unpromoted role evidence."""
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
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, out = Path(args.metrics), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    names = {'system': 'shared_compiler_repeats_audit_20260906.json',
        'bootstrap': 'bootstrap477_quality/decision.json',
        'role': 'sam2_role_probe_20260906.json',
        'prefix': 'sam2_prefix_invariance_20260906/result.json',
        'prefetch': 'verified_prefetch_residency_probe_20260906.json',
        'raw_motion': 'raw_cache_complete_234ec65/motion.json',
        'raw_state': 'raw_cache_complete_234ec65/state.json'}
    data, evidence = {}, []
    for key, name in names.items():
        path = root/name
        data[key] = json.loads(path.read_text())
        if data[key]['status'] != 'pass':
            raise ValueError(f'incomplete artifact: {path}')
        evidence.append({'id': key, 'path': str(path.resolve()),
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    def save(fig, name):
        fig.savefig(out/f'{name}.png', dpi=180)
        fig.savefig(out/f'{name}.pdf')
        plt.close(fig)
    def table(name, rows):
        with (out/f'{name}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    timing_rows = []
    for ax, group in zip(axes, data['system']['groups']):
        metric = group['summary']['end_to_end_s']
        for x, variant, color in [(0, 'old', '#a5b0bc'), (1, 'new', '#337db7')]:
            values = metric[f'{variant}_samples']
            ax.bar(x, statistics.median(values), color=color, width=.55)
            ax.scatter([x-.055, x+.055], values, color='black', s=15, zorder=3)
            for sample, value in enumerate(values):
                timing_rows.append({'prompt': group['prompt'], 'source': variant, 'repeat': sample,
                                    'complete_time_s': value, 'chronological_order': group['actual_successful_order']})
        ax.set_xticks([0, 1], ['Original compiler', 'Shared-union compiler'])
        ax.set(title=f"{group['prompt']}: {metric['reduction']*100:.2f}% lower\norder {group['actual_successful_order']}",
               ylabel='Complete video time (s)', ylim=(0, 230))
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Same routes, latent bytes, video bytes and H2D payload')
    fig.text(.02, .015, 'Two repeats per source; separate same-card prompts. State order changed after preserved pre-load failures. Model load separate.', fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, .94))
    save(fig, 'same_route_system_repeats')
    table('same_route_system_repeats', timing_rows)

    groups = data['bootstrap']['groups']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    rows = []
    for i, group in enumerate(groups):
        old, new = group['legacy'], group['bootstrap']
        rows.append({'cohort': group['cohort'], 'prompt': group['prompt'], 'seed': group['seed'],
            'full_lpips_delta': new['lpips_mean']-old['lpips_mean'],
            'late_lpips_delta': new['late_quarter_lpips_mean']-old['late_quarter_lpips_mean'],
            'latent_l2_delta': new['latent_error']['relative_l2']-old['latent_error']['relative_l2']})
    for ax, key, title in zip(axes, ('full_lpips_delta', 'late_lpips_delta'), ('Full video', 'Last quarter')):
        values = [r[key] for r in rows]
        ax.barh(range(len(rows)), values, color=['#b44a46' if v > 0 else '#367c60' for v in values])
        ax.axvline(0, color='black', linewidth=.7)
        ax.set_yticks(range(len(rows)), [f"{g['cohort']} {g['prompt']} / {str(g['seed'])[-2:]}" for g in groups])
        ax.set(title=title, xlabel='Bootstrap LPIPS minus Final (lower is better)')
    fig.suptitle('Long calibration rejects short-only promotion: 2/5 groups pass')
    fig.text(.02, .015, '15 technical passes. Same-device Dense controls; relative fidelity, not absolute semantic quality. No pooled hardware mean.', fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, .94))
    save(fig, 'bootstrap_long_nonpromotion')
    table('bootstrap_long_nonpromotion', rows)

    roles = data['role']['records']
    keys = list(roles[0]['metrics'])
    role_rows = [{'predictor': k,
        'soft_mask_mae': statistics.mean(r['metrics'][k]['soft_mask_mae'] for r in roles),
        'balanced_accuracy': statistics.mean(r['metrics'][k]['balanced_accuracy'] for r in roles)} for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    labels = ['Previous mask', 'Previous area', 'Raw Q', 'Raw Q + mask', 'RoPE Q', 'RoPE Q + mask']
    for ax, metric, title in zip(axes, ('soft_mask_mae', 'balanced_accuracy'), ('Soft-role error (lower)', 'Balanced accuracy (higher)')):
        ax.barh(range(len(keys)), [r[metric] for r in role_rows], color='#337db7')
        ax.set_yticks(range(len(keys)), labels)
        ax.set_title(title)
    fig.suptitle('Mask tracking works; the fixed Q-role formula adds no benefit here')
    fig.text(.02, .015, '40 correlated calls from one fixed-camera Dense state video; oracle historical masks, CPU summary diagnostic, not semantic ground truth.', fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, .94))
    save(fig, 'query_role_versus_spatial_baseline')
    table('query_role_versus_spatial_baseline', role_rows)

    prefetch = data['prefetch']['cohorts']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    prefetch_rows = []
    for kind, marker in [('motion', 'o'), ('state', 's')]:
        for predictor, label, color in [('previous_adjacent_layer', 'previous layer', '#bd704a'),
                                        ('same_layer_previous_chunk', 'previous chunk', '#337db7')]:
            row = prefetch[kind]['summary'][predictor]
            axes[0].scatter(row['aggregate_traffic_ratio_vs_whole_block'], row['aggregate_byte_recall'],
                            marker=marker, color=color, label=f'{kind}, {label}', s=65)
            prefetch_rows.append({'prompt': kind, 'predictor': predictor, **{k:row[k] for k in
                ('aggregate_byte_recall', 'aggregate_byte_precision', 'aggregate_traffic_ratio_vs_whole_block')}})
        residency = prefetch[kind]['per_layer_raw_residency']
        for layer, style in [('0', '-'), ('19', '--')]:
            budgets = sorted(map(int, residency[layer]))
            axes[1].plot(budgets, [residency[layer][str(b)]['aggregate_byte_hit_rate'] for b in budgets],
                         style, marker=marker, label=f'{kind}, L{layer}')
    axes[0].set(xlabel='Prefetch + miss bytes / non-prefetch Block64 bytes', ylabel='Useful byte recall',
                title='Temporal prediction uses fewer extra bytes', ylim=(0, .65))
    axes[1].set(xlabel='Raw cache per sampled layer (MiB)', ylabel='First-pass byte hit rate',
                title='Additional cross-chunk residency potential', ylim=(0, 1))
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.text(.02, .015, 'Causal trace replay only; exact24-token tails. No measured overlap/speedup, no extrapolation to unsampled layers or bounded CPU archive.', fontsize=8)
    fig.tight_layout(rect=(0, .055, 1, 1))
    save(fig, 'prefetch_and_residency')
    table('prefetch_and_residency', prefetch_rows)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    raw_rows = []
    for ax, kind in zip(axes, ('motion', 'state')):
        summary = data[f'raw_{kind}']['summary']
        modes = ('archive_runs', 'raw_cold', 'raw_warm')
        values = [summary[mode]['complete_s']*1000 for mode in modes]
        ax.bar(range(3), values, color=['#337db7', '#a5b0bc', '#bd704a'])
        ax.set_xticks(range(3), ['Runs\nuncached', 'Raw cache\ncold', 'Raw cache\nwarm'])
        ax.set(title=kind, ylabel='Complete materialization (ms)', ylim=(0, 780))
        for i, value in enumerate(values):
            ax.text(i, value+10, f'{value:.1f}', ha='center', fontsize=9)
        for mode in modes:
            raw_rows.append({'prompt': kind, 'mode': mode, **summary[mode]})
    fig.suptitle('Warm raw-cache prototype loses despite zero KV onload')
    fig.text(.02, .015, 'Same raw KV and route. Three measured repeats after warmup; preparation and token restoration included. No RoPE/Attention/video claim.', fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .94))
    save(fig, 'raw_cache_complete_cost_negative')
    table('raw_cache_complete_cost_negative', raw_rows)
    audit = {'status': 'pass', 'input_artifacts': evidence, 'figures': 5,
        'bootstrap_promoted': False, 'causal_role_promoted': False, 'prefetch_speedup_proven': False,
        'same_route_system_complete_time_reduction_observed': True, 'entire_plan_complete': False}
    (out/'report_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
