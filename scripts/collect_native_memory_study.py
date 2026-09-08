#!/usr/bin/env python3
"""Technical contracts only; quality requires separate own-source visual review."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_native_memory_study import ARMS


def validate_group(reports,gate=False):
    if set(reports)!=set(ARMS):raise ValueError('all five arms required')
    expected=[1,64,48,16,32] if gate else [1,128,48,44,80]
    source=list(range(8,16)) if gate else list(range(40,48))
    target=48 if gate else 96
    for arm,d in reports.items():
        if (d['status']!='pass' or d['latent_shape']!=expected
                or d['pixels']['frames']!=4*expected[1]-3 or d['sink_frames']!=8
                or d['attention_backend']!='native_FA2' or d['fallback_allowed']
                or d['local_frames']!=(128 if arm=='window128' else 32)):
            raise ValueError('technical/shape/capacity/backend mismatch: '+arm)
        if d['fixed_native_adaln_recipe']!=dict(num_warps=16,num_stages=1):
            raise ValueError('numerical recipe mismatch')
        if d['upstream_source_SHA']!='6b36d20ec6f7958d29d11a704dfa64611a9f2572':
            raise ValueError('upstream source mismatch')
        if d['strict_generator_load']['missing_keys'] or d['strict_generator_load']['unexpected_keys']:
            raise ValueError('incomplete released weights')
    for field in ('seed','cut_scenario','noise_sha256','runner_commit','assets_manifest_sha256'):
        if len({d[field] for d in reports.values()})!=1:raise ValueError('unmatched '+field)
    limited=[reports[a] for a in ARMS if a!='window128']
    for field in ('gpu','pre_return_latent_sha256'):
        if len({d[field] for d in limited})!=1:raise ValueError('limited-context pair differs: '+field)
    if not gate and len({d['gpu'] for d in reports.values()})!=1:raise ValueError('hardware changed within full group')
    if reports['none']['episode_memory_mode']!='none':raise ValueError('hidden baseline admission')
    if reports['none']['episode_memory']['ledger']['CPU_archive_peak_bytes']:raise ValueError('hidden baseline storage')
    if reports['window128']['episode_memory_mode'] is not None or 'episode_memory' in reports['window128']:
        raise ValueError('window control attached an episode hook')
    frame_tokens=expected[-1]*expected[-2]//4
    expected_kv=128*frame_tokens*24*128*2*2*30*2
    if reports['window128']['native_positive_and_negative_KV_bytes']!=expected_kv:
        raise ValueError('large window was not physically allocated as declared')
    for arm in ('global','global_one_chunk','shot'):
        memory=reports[arm]['episode_memory'];install=memory['installation']
        if (reports[arm]['episode_memory_mode']!='raw_reveal'
                or memory['capture']['source_frames']!=source or install['at_latent']!=target
                or not install['cache_metadata_unchanged'] or not memory['K_positions_preserved_not_rebased']):
            raise ValueError('wrong source/event/positions')
        lo,hi=install['admission_plan']['destination_token_range']
        if hi-lo!=8*frame_tokens or (lo<8*frame_tokens if arm=='shot' else lo!=0):
            raise ValueError('wrong destination range')
    global_d=reports['global'];ttl_d=reports['global_one_chunk']
    if global_d['first_return_latent_sha256']!=ttl_d['first_return_latent_sha256']:
        raise ValueError('lifetime intervention changed latent before expiry')
    memories=[reports[a]['episode_memory'] for a in ('global','global_one_chunk','shot')]
    for field in ('archive_D2H_payload_bytes','demand_H2D_payload_bytes'):
        if len({m['ledger'][field] for m in memories})!=1 or memories[0]['ledger'][field]<=0:
            raise ValueError('recall source payloads differ')
    g,t,s=memories
    if len({m['installation']['admission_plan_sha256'] for m in memories})!=3:
        raise ValueError('different activation policies need distinct admission SHA')
    expected_restore=t['ledger']['demand_H2D_payload_bytes']
    if (t['restore_after_frames']!=8 or t['restoration']['at_latent']!=target+8
            or not t['restoration']['original_pre_return_global_restored']
            or not t['restoration']['cache_metadata_unchanged']
            or t['ledger']['rollback_D2H_payload_bytes']!=expected_restore
            or t['ledger']['restore_H2D_payload_bytes']!=expected_restore
            or t['ledger']['CPU_archive_peak_bytes']!=g['ledger']['CPU_archive_peak_bytes']+expected_restore):
        raise ValueError('rollback identity/traffic/peak accounting incomplete')
    for m in (g,s):
        if m['restore_after_frames'] or m['restoration'] is not None:raise ValueError('unregistered expiry')
    return dict(status='pass',limited_context_pre_return_exact=True,global_vs_ttl_first_return_exact=True,
        window_control_has_own_trajectory=True,window_KV_bytes=expected_kv,
        same_recall_bytes_not_equal_rollback_cost=True,privileged_admission_not_autonomous=True,
        quality='not_scored_technical_gate_only' if gate else 'pending_own_source_absence_return_review')


def collect(root,gate=False):
    groups=[];all_rows=[]
    for lane in range(1 if gate else 4):
        reports={};rows=[]
        for arm in ARMS:
            source_lane=1 if gate and arm=='window128' else lane
            path=root/f'lane{source_lane}'/arm/'summary.json'
            try:
                d=json.loads(path.read_text());reports[arm]=d
                row=dict(lane=lane,arm=arm,status=d['status'],summary=str(path),
                    summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            except (OSError,ValueError) as error:
                row=dict(lane=lane,arm=arm,status='missing',summary=str(path),error=str(error))
            rows.append(row)
        try:
            contract=validate_group(reports,gate)
            if not gate:
                scenario=('generated_patchwork_toy_cut_revisit' if lane<2 else 'generated_bead_state_cut_revisit')
                if any(d['cut_scenario']!=scenario or d['seed']!=(20260913+lane%2) for d in reports.values()):
                    raise ValueError('wrong frozen group identity')
        except (KeyError,ValueError) as error:
            contract=dict(status='fail',error=str(error))
        groups.append(dict(lane=lane,contract=contract,cases=rows));all_rows+=rows
    return dict(status='pass' if all(g['contract']['status']=='pass' for g in groups) else 'fail',
        groups=groups,expected_cases=len(all_rows),technical_pass=sum(r['status']=='pass' for r in all_rows),
        technical_fail=sum(r['status']=='fail' for r in all_rows),missing=sum(r['status']=='missing' for r in all_rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--gate',action='store_true');p.add_argument('--output',type=Path)
    args=p.parse_args();result=collect(args.root,args.gate)
    output=args.output or args.root/'memory_study_terminal.json'
    with output.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='groups'}))
    if result['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
