#!/usr/bin/env python3
"""One GPU lane: source-locked placement/lifetime/capacity diagnostic.

Each child retains its own terminal report and log; failure never blocks the
remaining distinct cases. This launcher deliberately does not retry cases.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('none', 'global', 'global_one_chunk', 'shot', 'window128')


def arm_arguments(arm, gate=False):
    if arm not in ARMS:
        raise ValueError('unknown frozen memory arm')
    if arm == 'window128':
        result = ['--native-local-frames', '128']
    else:
        result = ['--episode-memory-mode', 'none' if arm == 'none' else 'raw_reveal',
                  '--episode-destination', 'shot' if arm == 'shot' else 'global']
        if arm == 'global_one_chunk':
            result += ['--episode-restore-after-frames', '8']
    if gate:
        result += ['--gate', '--episode-gate-layout']
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path, default=ROOT/'third_party/LongLive2')
    p.add_argument('--cut-scenario', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--arms', required=True)
    p.add_argument('--gate', action='store_true')
    args = p.parse_args()
    arms = args.arms.split(',')
    if len(set(arms)) != len(arms) or not set(arms) <= set(ARMS):
        raise ValueError('arms must be distinct members of the frozen study')
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for arm in arms:
        command = [sys.executable, str(ROOT/'scripts/run_longlive2_native_reference.py'),
                   '--assets', str(args.assets.resolve()), '--source', str(args.source.resolve()),
                   '--output', str((args.output/arm).resolve()), '--seed', str(args.seed),
                   '--cut-scenario', args.cut_scenario, '--fixed-adaln-warps', '16',
                   '--fixed-adaln-stages', '1', *arm_arguments(arm, args.gate)]
        with (args.output/f'{arm}.log').open('x') as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        summary_path = args.output/arm/'summary.json'
        record = dict(arm=arm, command=command, exit_code=code,
                      terminal_summary_exists=summary_path.exists())
        if not summary_path.exists():
            (args.output/arm).mkdir(exist_ok=True)
            summary_path.write_text(json.dumps(dict(status='fail', stage='child_process_exit',
                exit_code=code, log=f'../{arm}.log', partial_artifacts_preserved=True), indent=2)+'\n')
        record['status'] = json.loads(summary_path.read_text())['status']
        records.append(record)
        print(json.dumps(record), flush=True)
    with (args.output/'lane_terminal.json').open('x') as handle:
        json.dump(dict(cases=records, missing=0, attempts=len(arms)), handle, indent=2)
        handle.write('\n')
    if any(r['exit_code'] or r['status'] != 'pass' for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
