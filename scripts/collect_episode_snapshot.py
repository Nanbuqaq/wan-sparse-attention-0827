#!/usr/bin/env python3
"""Validate complete same-prefix, same-byte privileged KV-version interventions."""
import argparse
import hashlib
import json
from pathlib import Path


def validate(report,seed):
    if report['status']!='pass' or report['seed']!=seed or report['latent_frames']!=120:
        raise ValueError('snapshot identity/status mismatch')
    rows=report['variants']
    expected={'official_recache','snapshot_pre_recache','snapshot_post_recache'}
    if len(rows)!=3 or {r['variant'] for r in rows}!=expected:raise ValueError('three snapshot arms required')
    if len({(r['identity']['noise'],r['pre_return_latent_sha256']) for r in rows})!=1:
        raise ValueError('noise or pre-return trajectory differs')
    snapshots=[]
    for row in rows:
        if row['status']!='pass' or len(row['recache_events'])!=3:raise ValueError('incomplete native switches')
        if row['variant']=='official_recache':
            if row['history_H2D_bytes'] or row['archive_bytes']:raise ValueError('baseline archived history')
        else:
            capture,restore=row['snapshot_capture'],row['snapshot_restore']
            if capture['global_frames']!=list(range(42,48)) or restore['current_start']!=78:
                raise ValueError('privileged episode geometry differs')
            if not 0<row['archive_bytes']<=capture['budget_bytes'] or row['history_H2D_bytes']!=row['archive_bytes']:
                raise ValueError('snapshot storage/transfer not bounded and equal')
            if row['history_H2D_bytes']!=capture['D2H_bytes'] or not restore['attention_size_unchanged']:
                raise ValueError('copy accounting or local attention geometry differs')
            snapshots.append((row['history_H2D_bytes'],restore['local_slots_before_next_roll']))
    if snapshots[0]!=snapshots[1]:raise ValueError('pre/post version has a different byte/slot budget')
    return dict(status='pass',seed=seed,cases=3,gpu=report['gpu'],source_commit=report['source_commit'],
        same_pre_return_trajectory=True,same_snapshot_bytes_and_slots=True,privileged_diagnostic=True,
        quality='pending_own_identity_and_away_compliance_review')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--exit-codes',type=int,nargs=2,required=True);args=p.parse_args();rows=[]
    for lane,(seed,code) in enumerate(zip((20260909,20260910),args.exit_codes)):
        source=args.root/f'lane{lane}/summary.json'
        try:
            if code:raise ValueError(f'lane exit {code}')
            row=validate(json.loads(source.read_text()),seed)
            row.update(lane=lane,source_summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        except (OSError,KeyError,ValueError) as error:
            row=dict(lane=lane,status='fail',error=str(error),partial_artifacts_preserved=True)
        rows.append(row)
    report=dict(status='pass' if all(r['status']=='pass' for r in rows) else 'fail',lanes=rows,
        expected_cases=6,completed_cases=sum(r.get('cases',0) for r in rows),missing_unaccounted_lanes=0)
    with (args.root/'episode_snapshot_audit.json').open('x') as handle:json.dump(report,handle,indent=2);handle.write('\n')
    print(json.dumps(report))
    if report['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
