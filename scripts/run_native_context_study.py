#!/usr/bin/env python3
"""One local original-resolution lane, with a real capacity gate first."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
ARMS=('none','reset_reveal','reset_away','shot')


def arm_arguments(arm):
    if arm not in ARMS:raise ValueError('unknown context arm')
    args=['--cfg1-positive-cache-only','--episode-memory-mode',
          'none' if arm=='none' else ('raw_away' if arm=='reset_away' else 'raw_reveal')]
    if arm!='none':args+=['--episode-destination','shot']
    if arm.startswith('reset_'):args+=['--scene-context-reset']
    return args


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cut-scenario',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--order',default=','.join(ARMS));p.add_argument('--dry-run',action='store_true');args=p.parse_args()
    order=args.order.split(',')
    if set(order)!=set(ARMS) or len(order)!=4 or order[0]!='none':raise ValueError('exactly four arms, capacity baseline first')
    commands=[]
    for arm in order:
        commands.append(dict(arm=arm,command=[sys.executable,str(ROOT/'scripts/run_longlive2_native_reference.py'),
            '--assets',str(args.assets.resolve()),'--source',str(args.source.resolve()),
            '--output',str((args.output/arm).resolve()),'--seed',str(args.seed),
            '--cut-scenario',args.cut_scenario,'--fixed-adaln-warps','16','--fixed-adaln-stages','1',*arm_arguments(arm)]))
    if args.dry_run:print(json.dumps(dict(cases=commands,original_resolution=True,latent_frames=128,pixel_frames=509),indent=2));return
    args.output.mkdir(parents=True,exist_ok=False);records=[];capacity_failed=False
    for spec in commands:
        arm=spec['arm'];root=args.output/arm
        if capacity_failed:
            root.mkdir();d=dict(status='fail',stage='not_started_after_full_capacity_baseline_failure',
                               method_quality_evaluated=False,partial_artifacts_preserved=True)
            (root/'summary.json').write_text(json.dumps(d,indent=2)+'\n');code=None
        else:
            with (args.output/f'{arm}.log').open('x') as log:
                code=subprocess.run(spec['command'],stdout=log,stderr=subprocess.STDOUT).returncode
            if not (root/'summary.json').exists():
                root.mkdir(exist_ok=True)
                (root/'summary.json').write_text(json.dumps(dict(status='fail',stage='process_exit_without_report',exit_code=code),indent=2)+'\n')
            d=json.loads((root/'summary.json').read_text())
            if arm=='none' and (code or d['status']!='pass'):capacity_failed=True
        records.append(dict(arm=arm,exit_code=code,status=d['status'],stage=d.get('stage')))
        print(json.dumps(records[-1]),flush=True)
    with (args.output/'lane_terminal.json').open('x') as handle:json.dump(dict(cases=records,missing=0),handle,indent=2);handle.write('\n')
    if any(r['status']!='pass' for r in records):raise SystemExit(1)


if __name__=='__main__':main()
