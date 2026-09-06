#!/usr/bin/env python3
"""Same-GPU formal groups; each lane loads once and runs all frozen configs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_system_formal_suites import build
from adapters.longlive_sparse.case_identity import build_case_identity


def runtime_smoke(suites, calibration, source):
    """Exercise the exact frozen modes without generating a holdout frame."""
    result, expected = {}, []
    for config_id, suite in suites.items():
        method = suite['methods'][0]
        cases = []
        for prompt in calibration['candidates']:
            case = {**suite['cases'][0], **prompt, 'latent_frames': 39,
                    'seed': calibration['seeds'][0]}
            cases.append(case)
            identity = build_case_identity(commit=source, method=method,
                prompt_id=case['prompt_id'], prompt=case['prompt'], seed=case['seed'],
                latent_frames=39, history_density=case['history_density'], backend=case['backend'],
                rope_policy=case['rope_policy'], refresh_policy=case['refresh_policy'],
                system_identity=case['longlive_system'], method_params=suite['method_params'][method])
            expected.append({**identity, 'formal_config_id': config_id, 'method': method,
                'prompt_id': case['prompt_id'], 'seed': case['seed'], 'latent_frames': 39})
        result[config_id] = {**suite, 'cases': cases, 'formal_prompts_used': False,
                            'status': 'frozen_modes_runtime_regression_not_formal_quality'}
    return result, {'cases': expected, 'formal_prompts_used': False,
                    'scope': 'frozen_modes_runtime_regression_not_formal_quality'}


def regroup(suites, expected):
    config_ids = list(suites)
    first = suites[config_ids[0]]
    groups = [(c['prompt_id'], c['seed']) for c in first['cases']]
    output = {}
    for lane, group in enumerate(groups):
        ordered = config_ids[:]
        if lane%2:
            sparse = [c for c in ordered if c != 'rag_dense']
            ordered = ['rag_dense']+list(reversed(sparse))
        cases, methods = [], []
        for config_id in ordered:
            suite = suites[config_id]
            method = suite['methods'][0]
            candidate = next(c for c in suite['cases'] if (c['prompt_id'], c['seed']) == group)
            case = {**candidate, 'only_method': method, 'record_per_call': True,
                    'method_params': suite['method_params'][method]}
            cases.append(case)
            if method not in methods:
                methods.append(method)
        output[lane] = {**first, 'formal_config_id': 'matched_group', 'methods': methods,
            'method_params': {}, 'cases': cases, 'same_gpu_all_configs': True,
            'group_lane': lane, 'within_group_config_order': ordered}
    manifest = json.loads(json.dumps(expected))
    for case in manifest['cases']:
        case['lane'] = groups.index((case['prompt_id'], case['seed']))
    manifest['same_gpu_all_configs'] = True
    manifest['lanes'] = len(output)
    return output, manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--latent-frames', type=int, choices=(120, 240), required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--runtime-smoke', action='store_true')
    args = p.parse_args()
    source = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    suites, expected = build(holdout_path=ROOT/'configs/formal/system_holdout_prompts.json',
        method_freeze_path=ROOT/'configs/formal/system_method_freeze.json',
        method_params_path=ROOT/'configs/formal/method_params.json', latent_frames=args.latent_frames, commit=source)
    if args.runtime_smoke:
        calibration = json.loads((ROOT/'configs/system/profile_calibration_prompts.json').read_text())
        suites, expected = runtime_smoke(suites, calibration, source)
    lanes, expected = regroup(suites, expected)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    for lane, suite in lanes.items():
        (out/f'lane{lane}.json').write_text(json.dumps(suite, indent=2)+'\n')
    (out/'expected.json').write_text(json.dumps(expected, indent=2)+'\n')
    print(json.dumps({'source': source, 'lanes': len(lanes), 'cases': len(expected['cases'])}))


if __name__ == '__main__':
    main()
