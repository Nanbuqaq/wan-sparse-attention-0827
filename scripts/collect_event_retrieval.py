#!/usr/bin/env python3
"""Terminal audit only; automatic retrieval quality requires visual controls."""
import argparse
import hashlib
import json
from pathlib import Path


def validate(report,seed,control):
    expected={'scheduled_Dense','event_contrast'} if control else {'scheduled_Dense','event_cosine','event_contrast'}
    if (report['status']!='pass' or report['mode']!='event_probe' or report['seed']!=seed
            or report['latent_frames']!=120 or report.get('novel_control')!=control):
        raise ValueError('event case identity/status mismatch')
    rows=report['variants']
    if len(rows)!=len(expected) or {r['variant'] for r in rows}!=expected:raise ValueError('missing/duplicate event arm')
    if len({r['history_H2D_bytes'] for r in rows})!=1:raise ValueError('coarse retrieval changed raw KV byte budget')
    for row in rows:
        if row['status']!='pass':raise ValueError('incomplete event trajectory')
        if row['variant']!='scheduled_Dense':
            audit=row['event_retrieval']
            if not audit['past_episode_keys_only'] or audit['workload_role_labels_used'] or audit['predefined_anchor_interval_used']:
                raise ValueError('event method exceeded declared information boundary')
            if audit['first_override_latent'] is not None and not row['prefix_before_anchor_bitwise_equal']:
                raise ValueError('prefix differs before first retrieval intervention')
    return dict(status='pass',seed=seed,negative_control=control,gpu=report['gpu'],source_commit=report['source_commit'],
        cases=len(rows),same_raw_history_bytes=True,quality='pending_original_identity_and_negative_control_review')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--exit-codes',type=int,nargs=4,required=True);args=p.parse_args()
    design=[(20260909,None),(20260910,None),(20260909,'duck'),(20260910,'empty')]
    results=[]
    for lane,((seed,control),code) in enumerate(zip(design,args.exit_codes)):
        source=args.root/f'lane{lane}'/'summary.json'
        try:
            if code:raise ValueError(f'nonzero lane exit {code}')
            row=validate(json.loads(source.read_text()),seed,control)
            row.update(lane=lane,summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        except (OSError,ValueError,KeyError) as error:
            row=dict(lane=lane,status='fail',error=str(error),partial_results_preserved=True)
        results.append(row)
    report=dict(status='pass' if all(r['status']=='pass' for r in results) else 'fail',lanes=results,
        expected_cases=10,complete_cases=sum(r.get('cases',0) for r in results),missing_unaccounted_lanes=0,
        automatic_quality_success_not_inferred=True)
    with (args.root/'event_retrieval_audit.json').open('x') as handle:json.dump(report,handle,indent=2);handle.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='lanes'}))
    if report['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
