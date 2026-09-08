#!/usr/bin/env python3
"""Native cut-screen completeness; actual scene/memory feasibility is separate."""
import argparse
import hashlib
import json
from pathlib import Path


DESIGN=[(s,seed) for s in ('generated_patchwork_toy_cut_revisit','generated_bead_state_cut_revisit') for seed in (20260913,20260914)]


def validate(d,scenario,seed):
    if d['status']!='pass' or d['gate'] or d['cut_scenario']!=scenario or d['seed']!=seed:
        raise ValueError('native cut-screen identity/status differs')
    if d['latent_shape']!=[1,128,48,44,80] or d['local_frames']!=32 or d['sink_frames']!=8:
        raise ValueError('wrong full native geometry')
    if d['pixels']['frames']!=509 or d['attention_backend']!='native_FA2' or d['fallback_allowed']:
        raise ValueError('frame/backend/fallback gate failed')
    if d['expected_scene_cut_block_indices']!=[3,6,12] or [r['completed_latent'] for r in d['native_shot_pin_events']]!=[32,56,104]:
        raise ValueError('native scene transitions were not executed exactly once')
    if d['upstream_source_SHA']!='6b36d20ec6f7958d29d11a704dfa64611a9f2572':raise ValueError('native source changed')
    if d['strict_generator_load']['missing_keys'] or d['strict_generator_load']['unexpected_keys']:raise ValueError('incomplete weights')
    return dict(status='pass',scenario=scenario,seed=seed,gpu=d['gpu'],runner_commit=d['runner_commit'],
        native_shot_pin_events=d['native_shot_pin_events'],semantic_feasibility='pending_visual_review',
        Dense_only_no_new_memory_method=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--exit-codes',type=int,nargs=4,required=True)
    args=p.parse_args();rows=[]
    for lane,((scenario,seed),code) in enumerate(zip(DESIGN,args.exit_codes)):
        path=args.root/f'lane{lane}/summary.json'
        try:
            if code:raise ValueError(f'lane exit {code}')
            row=validate(json.loads(path.read_text()),scenario,seed)
            row.update(lane=lane,source_summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        except (OSError,KeyError,ValueError) as error:
            row=dict(lane=lane,status='fail',error=str(error),partial_artifacts_preserved=True)
        rows.append(row)
    result=dict(status='pass' if all(r['status']=='pass' for r in rows) else 'fail',lanes=rows,
        expected_cases=4,technical_pass=sum(r['status']=='pass' for r in rows),missing_unaccounted_lanes=0)
    with (args.root/'native_cut_screen_audit.json').open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps(result))
    if result['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
