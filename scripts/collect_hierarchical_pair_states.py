#!/usr/bin/env python3
"""Recover all successful and failed lanes, including failures before case init."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--exit-codes', nargs='+', type=int, required=True)
    args = p.parse_args()
    root = Path(args.root)
    inputs = []
    for lane in range(len(args.exit_codes)):
        state = root/f'lane{lane}/shard_0_states.json'
        if not state.is_file():
            state = root/f'lane{lane}_no_states.json'
            state.write_text(json.dumps({'cases': []})+'\n')
        inputs.extend(['--input', str(state)])
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here/'merge_case_states.py'), *inputs,
        '--expected', str(root/'control/expected.json'), '--fill-missing-reason', 'hierarchical lane failed before terminal state',
        '--output', str(root/'states.json')], check=True)
    subprocess.run([sys.executable, str(here/'audit_case_states.py'), '--expected', str(root/'control/expected.json'),
        '--states', str(root/'states.json'), '--output', str(root/'terminal_audit.json')], check=True)
    (root/'batch_exit.json').write_text(json.dumps({'lane_exit_codes': args.exit_codes,
        'status': 'pass' if not any(args.exit_codes) else 'fail'}, indent=2)+'\n')
    if any(args.exit_codes):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
