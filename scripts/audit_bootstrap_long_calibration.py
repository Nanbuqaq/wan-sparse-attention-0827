#!/usr/bin/env python3
"""Close long development cohorts without promoting mixed short-video gains."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

LEGACY = 'transfer_vaware_hybrid_history'
CANDIDATE = 'rope_bootstrap_ablation_history'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(4*1024**2), b''):
            digest.update(part)
    return digest.hexdigest()


def fidelity_checks(legacy, candidate):
    return {
        'full_lpips': candidate['lpips_mean'] <= legacy['lpips_mean'],
        'late_lpips': candidate['late_quarter_lpips_mean'] <= legacy['late_quarter_lpips_mean'],
        'latent_l2': candidate['latent_error']['relative_l2'] <= legacy['latent_error']['relative_l2'],
    }


def audit(workspace):
    results = workspace/'results'
    groups, evidence, cases_seen = [], [], set()
    cohorts = [
        ('h', results/'videos/bootstrap_calibration477_21393f3_h', 'recovered_states.json',
         'recovered_audit.json', [('motion', 20260916, 0), ('state', 20260916, 1), ('state', 20260917, 2)]),
        ('local', results/'videos/bootstrap477_b20a471_local', 'states.json',
         'terminal_audit.json', [('motion', 20260918, 0), ('state', 20260918, 1)]),
    ]
    for cohort, root, states_name, audit_name, targets in cohorts:
        terminal = json.loads((root/audit_name).read_text())
        if terminal['status'] != 'pass' or terminal['pass_cases'] != 3*len(targets) or terminal['errors']:
            raise ValueError(f'{cohort} terminal audit is not complete')
        states = json.loads((root/states_name).read_text())['cases']
        if len(states) != 3*len(targets):
            raise ValueError('unexpected cohort count')
        for kind, seed, lane in targets:
            triplet = [c for c in states if c['prompt_id'] == f'calibration_{kind}' and c['seed'] == seed]
            if len(triplet) != 3 or {c['method'] for c in triplet} != {'rag_dense', LEGACY, CANDIDATE}:
                raise ValueError('incomplete or duplicated matched triplet')
            by_method = {c['method']: c for c in triplet}
            for field in ('initial_noise_sha256', 'latent_frames', 'decoded_frames'):
                values = {c[field] for c in triplet}
                if len(values) != 1 or None in values:
                    raise ValueError(f'unmatched {field}')
            for field in ('prompt_sha256', 'system', 'rope_policy', 'backend'):
                if any(c['case_key'][field] != triplet[0]['case_key'][field] for c in triplet):
                    raise ValueError(f'unmatched config {field}')
            if triplet[0]['latent_frames'] != 120 or triplet[0]['decoded_frames'] != 477:
                raise ValueError('not a 477-frame cohort')
            for field in ('selected_history_tokens', 'candidate_history_tokens', 'transferred_bytes',
                          'candidate_transfer_bytes', 'history_pair_density'):
                if by_method[LEGACY][field] != by_method[CANDIDATE][field]:
                    raise ValueError(f'actual sparse budgets differ: {field}')
            if by_method[LEGACY]['history_pair_density'] != .25:
                raise ValueError('logical historical budget is not 25 percent')
            quality_path = results/f'metrics/bootstrap477_quality/{cohort}/{kind}_{seed}.json'
            quality = json.loads(quality_path.read_text())
            if quality['status'] != 'pass' or quality['reference_case'] != by_method['rag_dense']['id']:
                raise ValueError('wrong Dense quality reference')
            metrics = {c['method']: c for c in quality['rows']}
            for case in triplet:
                video = Path(case['video'])
                if video.parent.parent != root/f'lane{lane}':
                    raise ValueError('cross-lane reference contamination')
                if case['status'] != 'pass' or any(case[k] for k in ('failed_calls', 'nan_calls', 'fallback_calls')):
                    raise ValueError('technical fail or fallback')
                if sha(video) != case['video_sha256']:
                    raise ValueError('video artifact hash mismatch')
                if case['method'] != 'rag_dense':
                    metric = metrics[case['method']]
                    if (metric['case_id'] != case['id'] or metric['video_sha256'] != case['video_sha256']
                        or metric['reference_video_sha256'] != by_method['rag_dense']['video_sha256']):
                        raise ValueError('quality record uses another artifact')
                if case['id'] in cases_seen:
                    raise ValueError('case counted twice')
                cases_seen.add(case['id'])
            log = root/f'lane{lane}/runner.log'
            runtime = [json.loads(line[len('RUNTIME '):]) for line in log.read_text().splitlines()
                       if line.startswith('RUNTIME ')]
            if len(runtime) != 1:
                raise ValueError('one loaded same-device suite required')
            old, new = metrics[LEGACY], metrics[CANDIDATE]
            checks = fidelity_checks(old, new)
            selected = by_method[LEGACY]
            groups.append({'cohort': cohort, 'prompt': kind, 'seed': seed, 'lane': lane,
                'runtime': runtime[0], 'same_device_triplet': True,
                'initial_noise_sha256': triplet[0]['initial_noise_sha256'],
                'case_ids': {m: c['id'] for m, c in by_method.items()},
                'non_regression_checks': checks, 'gate_pass': all(checks.values()),
                'legacy': {k: old[k] for k in ('lpips_mean', 'late_quarter_lpips_mean', 'latent_error')},
                'bootstrap': {k: new[k] for k in ('lpips_mean', 'late_quarter_lpips_mean', 'latent_error')},
                'budget': {k: selected[k] for k in ('history_pair_density', 'history_transfer_density',
                    'global_executed_density', 'selected_history_tokens', 'transferred_bytes', 'candidate_transfer_bytes')},
                'quality_sha256': sha(quality_path), 'runner_log_sha256': sha(log)})
        evidence.extend({'path': str(root/name), 'sha256': sha(root/name)} for name in (states_name, audit_name))
    return {'status': 'pass', 'technical_cases': len(cases_seen), 'missing_cases': 0,
        'research_outcome': 'negative_for_promotion' if not all(g['gate_pass'] for g in groups) else 'development_gate_only',
        'formal_promotion': False, 'bootstrap_957_expansion': False,
        'groups_passed': sum(g['gate_pass'] for g in groups), 'groups_evaluated': len(groups),
        'groups': groups, 'evidence': evidence, 'pooled_quality_or_absolute_hardware_latency': False,
        'scope': 'relative_Dense_fidelity; absolute_semantic_quality_not_proven',
        'density_note': 'logical history is25%; transferred/candidate bytes across5 calls is5% with cache, not a5% route',
        'decision': 'retain all original short positives and long failures; do not tune on or open formal holdouts',
        'entire_plan_complete': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit(Path(args.workspace).resolve())
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('groups', 'evidence')}, indent=2))


if __name__ == '__main__':
    main()
