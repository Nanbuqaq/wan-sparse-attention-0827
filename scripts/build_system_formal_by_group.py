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
    args = p.parse_args()
    source = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    suites, expected = build(holdout_path=ROOT/'configs/formal/system_holdout_prompts.json',
        method_freeze_path=ROOT/'configs/formal/system_method_freeze.json',
        method_params_path=ROOT/'configs/formal/method_params.json', latent_frames=args.latent_frames, commit=source)
    lanes, expected = regroup(suites, expected)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    for lane, suite in lanes.items():
        (out/f'lane{lane}.json').write_text(json.dumps(suite, indent=2)+'\n')
    (out/'expected.json').write_text(json.dumps(expected, indent=2)+'\n')
    print(json.dumps({'source': source, 'lanes': len(lanes), 'cases': len(expected['cases'])}))


if __name__ == '__main__':
    main()
