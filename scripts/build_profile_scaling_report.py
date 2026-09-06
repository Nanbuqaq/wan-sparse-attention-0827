#!/usr/bin/env python3
"""Fixed-window latency, unbounded CPU archive, and measured kernel scope."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rows, sources = [], []
    for method in ('rag_dense', 'transfer_vaware_hybrid_history'):
        summary = json.loads((args.root/method/'summary.json').read_text())
        for index, length in enumerate((39, 120, 240), 1):
            record_path = args.root/method/f'lf{length}_record.json'
            trace_path = args.root/method/f'timeline.{index}.audit.json'
            record, trace = json.loads(record_path.read_text()), json.loads(trace_path.read_text())
            if summary['status'] != 'pass' or trace['status'] != 'pass' or not record['profiled_chunk_excluded_from_percentiles']:
                raise ValueError('audited profiles with traced chunk excluded required')
            rows.append({'method': method, 'gpu': summary['gpu'], 'latent_frames': length,
                'pixel_frames': record['decoded_frames'], 'chunk_p50_s': record['steady_chunk_span_p50_s'],
                'chunk_p95_s': record['steady_chunk_span_p95_s'], 'steady_samples': record['steady_unprofiled_chunk_samples'],
                'CPU_archive_KV_bytes': record['archive']['kv_bytes'], 'pinned_KV_bytes': record['archive']['pinned_kv_bytes'],
                'last_chunk_first_call_parent_s': trace['parent_wall_s'],
                'attention_kernel_fraction': trace['attention_kernel_active_fraction'],
                'complete_backend_fraction': trace['complete_attention_backend_wall_fraction'],
                'GPU_idle_in_profiled_parent_s': trace['GPU_idle_in_parent_s']})
            for path in (record_path, trace_path):
                sources.append({'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    if len({r['gpu'] for r in rows}) != 1:
        raise ValueError('cross-hardware absolute profiles cannot be pooled')
    archive = {r['latent_frames']: r['CPU_archive_KV_bytes'] for r in rows}
    slope = (archive[240]-archive[120])/120
    result = {'status': 'pass', 'rows': rows, 'sources': sources,
        'CPU_archive_bytes_per_additional_latent_120_to_240': slope,
        'bounded_GPU_and_pinned_buffers_do_not_bound_total_stream_storage': True,
        'last_chunk_first_call_not_all_denoising_average': True,
        'kernel_and_backend_fractions_are_nested_not_additive': True,
        '39_latent_percentiles_have_only_two_steady_samples': True,
        'diagnostic_generation_not_production_video_latency': True}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    with (args.output/'points.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    for method, label in [('rag_dense','Dense'),('transfer_vaware_hybrid_history','Final')]:
        selected = [r for r in rows if r['method'] == method]
        x = [r['latent_frames'] for r in selected]
        axes[0].plot(x, [r['chunk_p50_s'] for r in selected], marker='o', label=label+' p50')
        axes[0].plot(x, [r['chunk_p95_s'] for r in selected], linestyle='--', label=label+' p95')
    axes[0].set(xlabel='Latent frames', ylabel='Steady chunk span (s)', title='Fixed local/history GPU window')
    axes[0].legend(fontsize=8)
    axes[1].plot(sorted(archive), [archive[i]/1024**3 for i in sorted(archive)], marker='o')
    axes[1].set(xlabel='Latent frames', ylabel='CPU archive KV (GiB)', title='Still linear in stream length')
    x = np.arange(len(rows))
    axes[2].bar(x-.18, [100*r['attention_kernel_fraction'] for r in rows], .36, label='Attention kernels')
    axes[2].bar(x+.18, [100*r['complete_backend_fraction'] for r in rows], .36, label='Complete wrapper (contains kernels)')
    axes[2].set_xticks(x, [f"{'D' if r['method']=='rag_dense' else 'F'}{r['latent_frames']}" for r in rows], fontsize=8)
    axes[2].set(ylabel='% of measured parent', title='Nested scopes, not additive')
    axes[2].legend(fontsize=7)
    fig.suptitle(rows[0]['gpu']+'; profiled chunk excluded from steady percentiles')
    fig.savefig(args.output/'scaling_and_scope.png', dpi=150); plt.close(fig)
    print(json.dumps({'status': 'pass', 'records': len(rows), 'CPU_bytes_per_additional_latent': slope}))


if __name__ == '__main__':
    main()
