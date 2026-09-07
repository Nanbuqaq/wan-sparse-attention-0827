#!/usr/bin/env python3
"""Audit causal prototype reference videos against external Dense trajectories.

Latent distances and pixel diagnostics are not absolute generation quality.
The SDPA-null arm isolates backend-induced trajectory changes; this script
never labels prototype reference onload as an optimized online implementation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.history_cache import tensor_sha256
from adapters.longlive_sparse.offline_eval import output_error_metrics
from scripts.audit_system_video_comparison import load_case, sha
from scripts.build_video_review_storyboards import analyze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prototype-root', type=Path, required=True)
    parser.add_argument('--group-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    states = [json.loads(path.read_text()) | {'local_root': str(path.parent)}
              for path in args.group_root.rglob('case_state.json')]
    reports = []
    for kind in ('motion', 'state'):
        source = args.prototype_root/kind/'summary.json'
        summary = json.loads(source.read_text())
        if summary['status'] != 'pass' or len(summary['variants']) != 4:
            raise ValueError('four successful prototype reference arms required')
        matches = [r for r in states if r['method'] == 'rag_dense'
                   and r['prompt_id'] == summary['prompt']['prompt_id']
                   and r['case_key']['seed'] == summary['seed']
                   and r['case_key']['latent_frames'] == summary['latent_frames']]
        if len(matches) != 1:
            raise ValueError('exactly one compatible external Dense case required')
        dense = load_case(matches[0]['local_root'])
        if dense[3]['prompt'] != summary['prompt']['prompt']:
            raise ValueError('prompt text differs')
        references = {'rag_dense': dense[4]}
        rows = []
        for entry in summary['variants']:
            variant = entry['variant']
            root = args.prototype_root/kind/variant
            latent = torch.load(root/'latents.pt', map_location='cpu', weights_only=True)
            if tensor_sha256(latent) != entry['latent_sha256'] or not torch.isfinite(latent).all():
                raise ValueError('invalid latent/hash')
            if latent.shape != dense[4].shape:
                raise ValueError('incompatible latent shapes')
            references[variant] = latent
            row = dict(variant=variant, technical_status=entry['status'],
                       video=str(root/'video.mp4'), video_sha256=sha(root/'video.mp4'),
                       latent_sha256=entry['latent_sha256'],
                       initial_noise_sha256=entry['initial_noise_sha256'],
                       complete_wall_s=entry['complete_wall_s'],
                       onload_bytes=entry['base_runtime_reported_H2D_bytes']+entry['additional_reference_H2D_bytes'],
                       optimized_onload_claim=False, online_speed_Pareto_eligible=False)
            row['diagnostics'] = analyze(dict(id=kind+'__'+variant, video=row['video'],
                latent_frames=summary['latent_frames'], status=entry['status']), args.output, samples_per_quarter=16)
            rows.append(row)
        if len({r['initial_noise_sha256'] for r in rows}) != 1:
            raise ValueError('prototype reference arms do not share noise')
        dense_noise = dense[1].get('initial_noise_sha256')
        if dense_noise is not None and dense_noise != rows[0]['initial_noise_sha256']:
            raise ValueError('Dense/prototype noise differs')
        for row in rows:
            latent = references[row['variant']]
            row['fidelity_not_absolute_quality'] = {
                name: dict(full=output_error_metrics(ref, latent),
                           late_quarter=output_error_metrics(ref[:, -(summary['latent_frames']//4):],
                                                            latent[:, -(summary['latent_frames']//4):]))
                for name, ref in references.items() if name in ('rag_dense', 'legacy_final', 'sdpa_null')}
        dense_board = analyze(dict(id=kind+'__rag_dense', video=str(dense[0]/'video.mp4'),
            latent_frames=summary['latent_frames'], status='pass'), args.output, samples_per_quarter=16)
        reports.append(dict(kind=kind, prompt=summary['prompt'], seed=summary['seed'],
            source_summary_sha256=sha(source), source_commit=summary['source_commit'],
            dense_case=str(dense[0]), dense_case_sha256=sha(dense[0]/'case_state.json'),
            dense_initial_noise_hash_available=dense_noise is not None,
            dense_diagnostics=dense_board, variants=rows))
        print(json.dumps({'kind':kind, 'Dense_relative_L2': {r['variant']:
            r['fidelity_not_absolute_quality']['rag_dense']['full']['relative_l2'] for r in rows}}), flush=True)
    result = dict(status='pass', missing=0, prototype_cases=8, categories=reports,
                  visual_review='descriptive_AI_review_pending', absolute_quality_winner=None,
                  speed_claim=False, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
