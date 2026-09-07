#!/usr/bin/env python3
"""Collect every Dense feasibility lane without inferring semantic success."""
import argparse
import hashlib
import json
from pathlib import Path


def validate_report(report,scenario,seed):
    if report['status']!='pass' or report['mode']!='dense_screen' or report['seed']!=seed or report['latent_frames']!=120:
        raise ValueError('Dense screen identity/status mismatch')
    if report['scenario']['id']!=scenario or len(report['variants'])!=1:
        raise ValueError('Dense screen scenario/variant mismatch')
    row=report['variants'][0]
    if row['status']!='pass' or row['actual_fine_method']!='rag_dense' or row['nominal_history_density']!=1.:
        raise ValueError('all-history Dense feasibility case required')
    if len(row['schedule']['events'])!=4 or row['schedule']['future_generated_frames_read']:
        raise ValueError('event execution or causal boundary failed')
    return dict(status='pass',scenario=scenario,seed=seed,gpu=report['gpu'],source_commit=report['source_commit'],
        spec_sha256=report['spec_sha256'],video=row['video'],case_identity=row['case_identity_sha256'],
        generation_decode_encode_s=row['generation_decode_encode_s'],history_H2D_bytes=row['history_H2D_bytes'],
        semantic_validity='pending_manual_review',sparse_promotion=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--exit-codes',type=int,nargs=4,required=True);args=p.parse_args()
    design=[('development_canvas_state_revisit',20260909),('development_canvas_state_revisit',20260910),
            ('development_toy_identity_revisit',20260909),('development_toy_identity_revisit',20260910)]
    rows=[]
    for lane,((scenario,seed),exitcode) in enumerate(zip(design,args.exit_codes)):
        source=args.root/f'lane{lane}'/'summary.json'
        try:
            if exitcode:raise ValueError(f'lane exited {exitcode}')
            report=json.loads(source.read_text());row=validate_report(report,scenario,seed)
            row.update(lane=lane,summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        except (OSError,ValueError,KeyError) as error:
            row=dict(lane=lane,status='fail',error=str(error),exit_code=exitcode,partial_artifacts_preserved=True)
        rows.append(row)
    result=dict(status='pass' if all(r['status']=='pass' for r in rows) else 'fail',lanes=rows,
        expected_cases=4,technical_pass=sum(r['status']=='pass' for r in rows),missing_unaccounted_lanes=0,
        semantic_validity_and_sparse_promotion='pending_manual_review_not_inferred_from_exit_code')
    with (args.root/'dense_screen_audit.json').open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='lanes'}))
    if result['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
