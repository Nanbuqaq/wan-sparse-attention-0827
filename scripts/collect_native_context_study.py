#!/usr/bin/env python3
"""Audit original-resolution scene retirement, separately from semantic quality."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_native_context_study import ARMS


def validate_group(reports):
    if set(reports)!=set(ARMS):raise ValueError('four context arms required')
    for arm,d in reports.items():
        if (d['status']!='pass' or d['latent_shape']!=[1,128,48,44,80]
                or d['pixels']['frames']!=509 or d['local_frames']!=32 or d['sink_frames']!=8
                or d['attention_backend']!='native_FA2' or d['fallback_allowed']
                or d['native_KV_allocation_policy']!='CFG1_positive_only'
                or d['native_positive_and_negative_KV_bytes']!=10380902400):
            raise ValueError('incomplete/full-shape/common-capacity/backend mismatch: '+arm)
        if d['fixed_native_adaln_recipe']!=dict(num_warps=16,num_stages=1) or d['triton_version']!='3.2.0':
            raise ValueError('unmatched native numerical recipe')
        if d['scene_context_reset']!=arm.startswith('reset_'):raise ValueError('wrong reset policy')
    for field in ('seed','cut_scenario','noise_sha256','pre_return_latent_sha256','runner_commit','assets_manifest_sha256','gpu'):
        if len({d[field] for d in reports.values()})!=1:raise ValueError('unmatched '+field)
    baseline=reports['none']['episode_memory']
    if baseline['mode']!='none' or baseline['ledger']['CPU_archive_peak_bytes']:raise ValueError('hidden baseline memory')
    for arm in ('shot','reset_reveal','reset_away'):
        memory=reports[arm]['episode_memory'];install=memory['installation'];plan=install['admission_plan']
        expected=list(range(56,64)) if arm=='reset_away' else list(range(40,48))
        if (memory['capture']['source_frames']!=expected or plan['source_frames']!=expected
                or plan['destination_token_range']!=[7040,14080] or install['at_latent']!=96
                or not memory['K_positions_preserved_not_rebased']):raise ValueError('source/position/event mismatch')
        if memory['ledger']['demand_H2D_payload_bytes']!=2595225600 or memory['ledger']['CPU_archive_peak_bytes']!=2595225600:
            raise ValueError('recall byte budget differs')
        if arm=='shot':
            if not install['cache_metadata_unchanged']:raise ValueError('ordinary shot changed endpoints')
        else:
            transition=install['cache_metadata_transition'];before=transition['before'];after=transition['after']
            if before!=dict(global_end_index=84480,local_end_index=28160,pinned_start=7040,pinned_len=7040):
                raise ValueError('unexpected pre-return cache geometry')
            if after!=dict(before,local_end_index=14080):raise ValueError('reset changed absolute clock or global/shot positions')
            expected_shapes=[dict(query_start_latent=f,Q=7040,K=21120 if f==96 else 28160,calls=150) for f in (96,104,112,120)]
            if memory['observed_return_attention_shapes']!=expected_shapes:raise ValueError('actual execution did not use declared logical context')
    return dict(status='pass',actual_pre_return_latents_require_file_audit=True,
        same_source_recall_bytes=True,same_allocated_positive_KV=True,first_return_K_24_then_32=True,
        privileged_admission_not_autonomous=True,single_seed_development_not_holdout=True,
        quality='pending_own_source_absence_return_review')


def collect(root):
    groups=[];rows=[]
    for lane in range(2):
        reports={};cases=[]
        for arm in ARMS:
            path=root/f'lane{lane}'/arm/'summary.json'
            try:
                d=json.loads(path.read_text());reports[arm]=d
                row=dict(arm=arm,status=d['status'],stage=d.get('stage'),summary=str(path),
                         summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            except (OSError,ValueError) as error:row=dict(arm=arm,status='missing',error=str(error))
            cases.append(row)
        try:
            contract=validate_group(reports)
            scenario='generated_patchwork_toy_cut_revisit' if lane==0 else 'generated_bead_state_cut_revisit'
            if any(d['seed']!=20260913 or d['cut_scenario']!=scenario for d in reports.values()):raise ValueError('wrong frozen identity')
        except (KeyError,ValueError) as error:contract=dict(status='fail',error=str(error))
        groups.append(dict(lane=lane,contract=contract,cases=cases));rows+=cases
    return dict(status='pass' if all(g['contract']['status']=='pass' for g in groups) else 'fail',groups=groups,
        expected_cases=8,technical_pass=sum(r['status']=='pass' for r in rows),technical_fail=sum(r['status']=='fail' for r in rows),
        missing=sum(r['status']=='missing' for r in rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    result=collect(args.root)
    with (args.root/'context_study_terminal.json').open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='groups'}))
    if result['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
