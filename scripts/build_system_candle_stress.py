#!/usr/bin/env python3
"""Prespecified optional difficult prompt, post-formal descriptive stress only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.case_identity import build_case_identity
from scripts.build_system_formal_suites import build
from scripts.build_system_formal_by_group import regroup


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    source = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    templates, _ = build(holdout_path=ROOT/'configs/formal/system_holdout_prompts.json',
        method_freeze_path=ROOT/'configs/formal/system_method_freeze.json',
        method_params_path=ROOT/'configs/formal/method_params.json', latent_frames=120, commit=source)
    prompt_path = ROOT/'configs/prompts/dense_candidates.json'
    data = json.loads(prompt_path.read_text())
    candidates = data if isinstance(data, list) else data['candidates']
    prompt = next(c for c in candidates if c['prompt_id'] == 'state_melting_candle')
    suites, expected = {}, []
    for config_id, suite in templates.items():
        method, cases = suite['methods'][0], []
        params = suite['method_params'][method]
        for seed in (20260911, 20260912):
            case = {**suite['cases'][0], **prompt, 'seed': seed, 'method_params': params}
            cases.append(case)
            identity = build_case_identity(commit=source, method=method, prompt_id=case['prompt_id'],
                prompt=case['prompt'], seed=seed, latent_frames=120, history_density=case['history_density'],
                backend=case['backend'], rope_policy=case['rope_policy'], refresh_policy=case['refresh_policy'],
                system_identity=case['longlive_system'], method_params=params)
            expected.append({**identity, 'method': method, 'formal_config_id': config_id,
                'prompt_id': case['prompt_id'], 'seed': seed, 'latent_frames': 120})
        suites[config_id] = {**suite, 'cases': cases, 'formal_prompts_used': False,
            'status': 'frozen_optional_candle_stress_after_formal477',
            'exclude_from_formal_mean_pareto_selection': True,
            'source_prompt_sha256': hashlib.sha256(prompt_path.read_bytes()).hexdigest()}
    grouped, manifest = regroup(suites, {'cases': expected,
        'scope': 'optional_post_formal_descriptive_stress_not_independent_holdout',
        'exclude_from_formal_mean_pareto_selection': True})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for lane, suite in grouped.items():
        (args.output_dir/f'lane{lane}.json').write_text(json.dumps(suite, indent=2)+'\n')
    (args.output_dir/'expected.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'source': source, 'cases': len(expected), 'lanes': len(grouped),
                      'exclude_from_formal_mean_pareto_selection': True}))


if __name__ == '__main__':
    main()
