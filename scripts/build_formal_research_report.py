#!/usr/bin/env python3
"""Audited formal tables and figures; full cost and fidelity remain separate."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(values):
    return {'min': min(values), 'median': statistics.median(values), 'max': max(values),
            'geometric_mean': math.exp(statistics.mean(map(math.log, values)))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--system-audit', type=Path, required=True)
    p.add_argument('--quality-root', type=Path, required=True)
    p.add_argument('--visual-audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    system = json.loads(args.system_audit.read_text())
    visual = json.loads(args.visual_audit.read_text())
    if system['status'] != 'pass' or visual['status'] != 'pass' or system['missing'] or visual['missing']:
        raise ValueError('complete equivalence and visual audits required')
    quality_paths = sorted(args.quality_root.glob('lane*/*/quality.json'))
    quality = {}
    for path in quality_paths:
        data = json.loads(path.read_text())
        if data['status'] != 'pass' or data.get('metric_input_protocol_version') != 2:
            raise ValueError('only canonical v2 scores are valid')
        for r in data['rows']:
            if r['case_id'] in quality:
                raise ValueError('duplicate quality case')
            quality[r['case_id']] = r
    reviewed = {r['case_id']: r for r in visual['rows']}
    rows, pairs = [], []
    for group in system['groups']:
        old, new, dense = (group['cases'][k] for k in ('legacy_final', 'legacy_final_system', 'rag_dense'))
        for config, case in group['cases'].items():
            fidelity = quality[case['case_id']]
            review = reviewed[case['case_id']]
            phase = case['wall_breakdown']
            pipeline_s = phase['inference_s']+phase['deferred_vae_s']
            tail_fields = ('finite_latent_save_RGB_conversion_hash_s', 'encoding_s', 'preview_frame_verification_s')
            residual = case['end_to_end_s']-pipeline_s-sum(phase[f] for f in tail_fields)
            if residual < -1e-6:
                raise ValueError('outer wall phases overlap or exceed complete time')
            rows.append({'prompt': group['prompt'], 'seed': group['seed'], 'latent_frames': group['latent_frames'],
                'config': config, 'case_id': case['case_id'], 'end_to_end_s': case['end_to_end_s'],
                'model_load_s_total': case['model_load_s_total'],
                'complete_with_one_third_loaded_model_s': case['end_to_end_s']+case['model_load_s_total']/3,
                'pipeline_with_vae_s': pipeline_s,
                'raw_pixel_and_latent_artifacts_s': phase['finite_latent_save_RGB_conversion_hash_s'],
                'encoding_s': phase['encoding_s'], 'preview_verify_s': phase['preview_frame_verification_s'],
                'setup_and_remaining_audit_s': max(0., residual),
                'normalized_complete_time_vs_dense': case['end_to_end_s']/dense['end_to_end_s'],
                'history_pair_density': case['history_pair_density'], 'history_transfer_density': case['history_transfer_density'],
                'global_executed_density': case['global_executed_density'], 'KV_H2D_bytes': case['transferred_bytes'],
                'CPU_archive_KV_bytes': case['archive_storage']['kv_bytes'], 'peak_GPU_allocated_GiB': case['peak_allocated_gb'],
                'lpips_mean': fidelity['lpips_mean'], 'late_quarter_lpips_mean': fidelity['late_quarter_lpips_mean'],
                'ssim_mean': fidelity['ssim_mean'], 'psnr_mean_cap100': fidelity['psnr_mean'],
                'late_quarter_visual_1to5': review['grade']['late_quarter_quality_1to5'],
                'category_completion_0to2': review['grade']['category_completion_0to2']})
        if quality[old['case_id']]['latent_sha256'] != quality[new['case_id']]['latent_sha256']:
            raise ValueError('lossless systems did not share a canonical render')
        old_phase, new_phase = old['wall_breakdown'], new['wall_breakdown']
        pairs.append({'prompt': group['prompt'], 'seed': group['seed'], 'latent_frames': group['latent_frames'],
            'system_speedup_complete': old['end_to_end_s']/new['end_to_end_s'],
            'system_speedup_pipeline_with_vae': (old_phase['inference_s']+old_phase['deferred_vae_s'])/(new_phase['inference_s']+new_phase['deferred_vae_s']),
            'Final_system_speedup_vs_optimized_Dense': dense['end_to_end_s']/new['end_to_end_s'],
            'KV_H2D_reduction': 1-new['transferred_bytes']/old['transferred_bytes'],
            'same_full_Final_output': True})
    if len(rows) != len(quality) or set(reviewed) != set(quality):
        raise ValueError('missing metrics/visual results in report')
    result = {'status': 'pass', 'cases': len(rows), 'independent_prompt_seed_groups': len(pairs), 'missing': 0,
        'system_speedup_complete': summarize([r['system_speedup_complete'] for r in pairs]),
        'system_speedup_pipeline_with_vae': summarize([r['system_speedup_pipeline_with_vae'] for r in pairs]),
        'Final_system_speedup_vs_optimized_Dense': summarize([r['Final_system_speedup_vs_optimized_Dense'] for r in pairs]),
        'paired_comparisons': pairs, 'rows': rows,
        'no_repeated_run_confidence_interval_claim': True, 'formal_no_retuning': True,
        'Dense_fidelity_not_absolute_quality': True, 'no_better_than_Dense_quality_claim': True,
        'model_load_separate_and_also_amortized': True,
        'system_quality_inherited_by_exact_pixels_not_independent_samples': True,
        'sources': [{'path': str(p.resolve()), 'sha256': sha(p)} for p in
                    [args.system_audit, args.visual_audit, *quality_paths]]}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    for name, data in (('cases.csv', rows), ('pairs.csv', pairs)):
        with (args.output/name).open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    groups = [(r['prompt'], r['seed']) for r in pairs]
    configs = ['rag_dense', 'legacy_final', 'legacy_final_system']
    lookup = {(r['prompt'], r['seed'], r['config']): r for r in rows}
    labels = [f"{p.replace('identity_mars_', '').replace('state_blue_canvas_', '').replace('human_', '').replace('fast_', '')}\n{s}" for p,s in groups]
    fig, ax = plt.subplots(figsize=(13, 5), layout='constrained')
    for j, config in enumerate(configs):
        values = [lookup[p,s,config]['end_to_end_s'] for p,s in groups]
        ax.bar(np.arange(len(groups))+(j-1)*.24, values, width=.24, label=config)
    ax.set_xticks(range(len(groups)), labels, fontsize=8)
    ax.set(ylabel='Complete seconds (generation + VAE + artifact/audit)', title='Frozen matched groups; model load separately reported')
    ax.legend(fontsize=9)
    fig.savefig(args.output/'complete_latency.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    for ax, prompt in zip(axes.flatten(), sorted({r['prompt'] for r in rows})):
        for seed in sorted({r['seed'] for r in rows if r['prompt'] == prompt}):
            a, b = lookup[prompt,seed,'legacy_final'], lookup[prompt,seed,'legacy_final_system']
            x0,x1 = a['normalized_complete_time_vs_dense'],b['normalized_complete_time_vs_dense']
            y = a['lpips_mean']
            ax.plot([x0,x1], [y,y], marker='o', label=str(seed))
            ax.annotate('', xy=(x1,y), xytext=(x0,y), arrowprops={'arrowstyle':'->'})
        ax.scatter([1.], [0.], marker='D', color='black', label='matched Dense reference')
        ax.set(title=prompt, xlabel='Complete time / same-group optimized Dense', ylabel='Canonical LPIPS vs Dense')
        ax.legend(fontsize=7)
    fig.savefig(args.output/'lossless_system_shift.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 5), layout='constrained')
    fields = ['pipeline_with_vae_s','raw_pixel_and_latent_artifacts_s','encoding_s','preview_verify_s','setup_and_remaining_audit_s']
    selected = [lookup[p,s,c] for p,s in groups for c in ('legacy_final','legacy_final_system')]
    bottom = np.zeros(len(selected))
    for field in fields:
        values = [r[field] for r in selected]
        ax.bar(range(len(selected)), values, bottom=bottom, label=field)
        bottom += values
    ax.set_xticks(range(len(selected)), [f"{i//2}: {'legacy' if i%2==0 else 'system'}" for i in range(len(selected))], rotation=60, fontsize=7)
    ax.set(ylabel='Seconds', title='Nonoverlapping outer wall phases; artifact tail is not attributed to KV execution')
    ax.legend(fontsize=7)
    fig.savefig(args.output/'complete_wall_phases.png', dpi=150); plt.close(fig)
    lines = ['# Frozen formal cohort report', '',
        'Matched configurations; no formal tuning, no new admission, no claim of better absolute quality than Dense.', '',
        '| Prompt | Seed | Dense s | Final legacy s | Final system s | System speedup | LPIPS | Late LPIPS |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for pair in pairs:
        p,s = pair['prompt'],pair['seed']
        d,a,b = (lookup[p,s,c] for c in configs)
        lines.append(f"|{p}|{s}|{d['end_to_end_s']:.3f}|{a['end_to_end_s']:.3f}|{b['end_to_end_s']:.3f}|{pair['system_speedup_complete']:.3f}x|{b['lpips_mean']:.4f}|{b['late_quarter_lpips_mean']:.4f}|")
    lines += ['', 'System variants have identical full latents, routes and preencode pixels. Complete wall includes',
        'generation, VAE and artifact/audit work. Inference+VAE and startup are separately retained, not used',
        'to erase slow artifact tails. Different prompt/seed groups are not timing repetitions.', '',
        'Visual audit is unblinded assistant inspection of all quarter boards and selected details, not a human',
        'preference study. Same-Dense fidelity does not establish subject, material or state correctness.', '',
        '![Complete latency](complete_latency.png)', '![Paired lossless system shift](lossless_system_shift.png)',
        '![Full wall accounting](complete_wall_phases.png)', '']
    (args.output/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','paired_comparisons','sources')}, indent=2))


if __name__ == '__main__':
    main()
