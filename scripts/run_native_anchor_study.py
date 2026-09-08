#!/usr/bin/env python3
"""Two same-byte source controls for one explicit initial-anchor policy."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--policy',choices=('source_only','source_repeat'),required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--cut-scenario',required=True);p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();modes=['raw_reveal','raw_away'] if args.policy=='source_only' else ['raw_away','raw_reveal'];commands=[]
    for mode in modes:
        commands.append(dict(mode=mode,command=[sys.executable,str(ROOT/'scripts/run_longlive2_native_reference.py'),
            '--assets',str(args.assets.resolve()),'--source',str(args.source.resolve()),'--output',str((args.output/mode).resolve()),
            '--cut-scenario',args.cut_scenario,'--seed',str(args.seed),'--cfg1-positive-cache-only',
            '--episode-memory-mode',mode,'--episode-destination','global','--scene-context-reset','--initial-anchor-policy',args.policy,
            '--fixed-adaln-warps','16','--fixed-adaln-stages','1']))
    plan=dict(policy=args.policy,commands=commands,privileged_admission_not_autonomous=True,
        physical_window=32,source_unique_latents=8,source_multiplicity=1 if args.policy=='source_only' else 2)
    if args.dry_run:print(json.dumps(plan,indent=2));return
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');rows=[]
    for entry in commands:
        with (args.output/f'{entry["mode"]}.log').open('x') as log:
            code=subprocess.run(entry['command'],stdout=log,stderr=subprocess.STDOUT).returncode
        path=args.output/entry['mode']/'summary.json'
        if not path.exists():
            path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(dict(status='fail',stage='child_exit_without_report',exit_code=code),indent=2)+'\n')
        row=dict(mode=entry['mode'],status=json.loads(path.read_text())['status'],exit_code=code);rows.append(row);print(json.dumps(row),flush=True)
    (args.output/'terminal.json').write_text(json.dumps(dict(cases=rows,missing=0),indent=2)+'\n')
    if any(r['status']!='pass' or r['exit_code'] for r in rows):raise SystemExit(1)


if __name__=='__main__':main()
