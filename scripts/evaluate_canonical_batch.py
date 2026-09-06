#!/usr/bin/env python3
"""One serial GPU lane of locked canonical groups; isolate evaluation failures."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--states', type=Path, required=True)
    p.add_argument('--expected', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--linear-weights', required=True)
    p.add_argument('--trunk-weights', required=True)
    p.add_argument('--lane', type=int, required=True)
    p.add_argument('--lanes', type=int, default=2)
    args = p.parse_args()
    if not 0 <= args.lane < args.lanes:
        raise ValueError('invalid evaluation lane')
    # Fail before loading VAE/GPU state if an operator mistypes an input path.
    protocol_path = Path(__file__).resolve().parents[1]/'configs/quality/lpips_alex_v0p1.json'
    protocol = json.loads(protocol_path.read_text())
    for field, path in (('linear_weights', args.linear_weights), ('trunk_weights', args.trunk_weights)):
        with Path(path).open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if digest != protocol[field]['sha256']:
            raise ValueError(f'{field} differs from the frozen quality protocol')
    manifest = json.loads(args.expected.read_text())
    groups = sorted({(c['prompt_id'], c['seed'], c['latent_frames']) for c in manifest['cases']})[args.lane::args.lanes]
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for prompt, seed, length in groups:
        name = f'{prompt}__s{seed}__lf{length}'
        directory = args.output/name
        log = args.output/f'{name}.log'
        cmd = [sys.executable, str(Path(__file__).with_name('evaluate_canonical_video_quality.py')),
            '--states', str(args.states), '--expected', str(args.expected), '--prompt', prompt, '--seed', str(seed),
            '--latent-frames', str(length), '--linear-weights', args.linear_weights, '--trunk-weights', args.trunk_weights,
            '--output', str(directory)]
        with log.open('x') as handle:
            code = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT).returncode
        result_path = directory/'quality.json'
        passing = code == 0 and result_path.is_file()
        records.append({'prompt': prompt, 'seed': seed, 'latent_frames': length,
            'status': 'pass' if passing else 'fail', 'exit_code': code, 'quality': str(result_path.resolve()) if passing else None,
            'sha256': hashlib.sha256(result_path.read_bytes()).hexdigest() if passing else None,
            'log': str(log.resolve()), 'failure_scope': None if passing else 'evaluation_not_generation'})
        print(json.dumps(records[-1]), flush=True)
    result = {'status': 'pass' if all(r['status'] == 'pass' for r in records) else 'fail',
        'lane': args.lane, 'groups': records, 'missing': 0,
        'expected_sha256': hashlib.sha256(args.expected.read_bytes()).hexdigest()}
    (args.output/'terminal.json').write_text(json.dumps(result, indent=2)+'\n')
    if result['status'] != 'pass':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
