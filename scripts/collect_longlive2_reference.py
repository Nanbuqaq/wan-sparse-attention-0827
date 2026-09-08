#!/usr/bin/env python3
"""Four native LongLive2 controls; technical completeness is not semantic success."""
import argparse
import hashlib
import json
from pathlib import Path


def validate(d,seed,control):
    if d['status']!='pass' or d['seed']!=seed or d['control']!=control or d['gate']:
        raise ValueError('native reference case identity/status mismatch')
    if d['latent_shape']!=[1,128,48,44,80] or d['local_frames']!=32 or d['sink_frames']!=8:
        raise ValueError('native full shape/context changed')
    if d['pixel_frames']!=509 or d['pixels']['frames']!=509:raise ValueError('native video frame count differs')
    if d['upstream_source_SHA']!='6b36d20ec6f7958d29d11a704dfa64611a9f2572' or d['attention_backend']!='native_FA2' or d['fallback_allowed']:
        raise ValueError('source/backend/fallback mismatch')
    if d['strict_generator_load']['missing_keys'] or d['strict_generator_load']['unexpected_keys']:
        raise ValueError('incomplete native released weights')
    return dict(status='pass',seed=seed,control=control,gpu=d['gpu'],runner_commit=d['runner_commit'],
        source_commit=d['upstream_source_SHA'],pixel_frames=509,quality='pending_descriptive_review',
        cross_backbone_absolute_speed_not_comparable=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--exit-codes',type=int,nargs=4,required=True)
    args=p.parse_args();rows=[]
    for lane,((seed,control),code) in enumerate(zip(((20260909,None),(20260910,None),(20260909,'duck'),(20260910,'empty')),args.exit_codes)):
        source=args.root/f'lane{lane}/summary.json'
        try:
            if code:raise ValueError(f'lane exit {code}')
            row=validate(json.loads(source.read_text()),seed,control)
            row.update(lane=lane,summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        except (OSError,KeyError,ValueError) as error:
            row=dict(lane=lane,status='fail',error=str(error),partial_artifacts_preserved=True)
        rows.append(row)
    result=dict(status='pass' if all(r['status']=='pass' for r in rows) else 'fail',lanes=rows,
        expected_cases=4,technical_pass=sum(r['status']=='pass' for r in rows),missing_unaccounted_lanes=0)
    with (args.root/'native_reference_audit.json').open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps(result))
    if result['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
