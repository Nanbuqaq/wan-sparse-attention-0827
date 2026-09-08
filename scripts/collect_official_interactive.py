#!/usr/bin/env python3
"""Audit the eight-case native recache control; semantic quality is separate."""
import argparse
import hashlib
import json
from pathlib import Path


def validate(report,seed,control):
    if report['status']!='pass' or report['seed']!=seed or report['latent_frames']!=120 or report.get('novel_control')!=control:
        raise ValueError('interactive control identity/status mismatch')
    rows=report['variants']
    if len(rows)!=2 or {r['variant'] for r in rows}!={'cross_only','official_recache'}:
        raise ValueError('both interactive control arms required')
    for row in rows:
        if row['status']!='pass' or row['history_H2D_bytes'] or row['archive_bytes']:
            raise ValueError('local control used historical archive or failed')
        if len(row['recache_events'])!=3:raise ValueError('three prompt-switch branches required')
    return dict(status='pass',seed=seed,negative_control=control,gpu=report['gpu'],source_commit=report['source_commit'],
        upstream_interactive_sha256=report['upstream_interactive_sha256'],cases=2,
        quality='pending_own_identity_absence_and_instruction_review')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--exit-codes',type=int,nargs=4,required=True);args=p.parse_args()
    design=[(20260909,None),(20260910,None),(20260909,'duck'),(20260910,'empty')]
    rows=[]
    for lane,((seed,control),code) in enumerate(zip(design,args.exit_codes)):
        source=args.root/f'lane{lane}'/'summary.json'
        try:
            if code:raise ValueError(f'lane exit {code}')
            row=validate(json.loads(source.read_text()),seed,control)
            row.update(lane=lane,summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        except (ValueError,KeyError,OSError) as error:
            row=dict(lane=lane,status='fail',error=str(error),partial_artifacts_preserved=True)
        rows.append(row)
    result=dict(status='pass' if all(r['status']=='pass' for r in rows) else 'fail',lanes=rows,
        expected_cases=8,completed_cases=sum(r.get('cases',0) for r in rows),missing_unaccounted_lanes=0)
    with (args.root/'official_interactive_audit.json').open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='lanes'}))
    if result['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
