#!/usr/bin/env python3
"""Admission/storage factorial gates, independent of later semantic review."""
import argparse
import hashlib
import json
from pathlib import Path

MODES=('none','raw_reveal','raw_away','log_reveal')
DESIGN=[(s,seed) for s in ('generated_patchwork_toy_cut_revisit','generated_bead_state_cut_revisit') for seed in (20260913,20260914)]


def validate_group(reports,*,gate=False):
    if set(reports)!=set(MODES):raise ValueError('four episode modes required')
    expected=[1,64,48,16,32] if gate else [1,128,48,44,80]
    source=list(range(8,16)) if gate else list(range(40,48))
    wrong=list(range(24,32)) if gate else list(range(56,64))
    for mode,d in reports.items():
        if d['status']!='pass' or d['episode_memory_mode']!=mode or d['latent_shape']!=expected:
            raise ValueError('episode identity/shape/status mismatch')
        if d['local_frames']!=32 or d['sink_frames']!=8 or d['attention_backend']!='native_FA2' or d['fallback_allowed']:
            raise ValueError('native context/backend mismatch')
        if d['pixels']['frames']!=4*expected[1]-3:raise ValueError('episode video truncated')
    if len({(d['seed'],d['cut_scenario'],d['gpu'],d['noise_sha256'],d['pre_return_latent_sha256']) for d in reports.values()})!=1:
        raise ValueError('matched initial identity/hardware/pre-return prefix required')
    raw,log,away=reports['raw_reveal'],reports['log_reveal'],reports['raw_away']
    if raw['latent_sha256']!=log['latent_sha256'] or raw['pixels']['raw_RGB_sha256']!=log['pixels']['raw_RGB_sha256']:
        raise ValueError('raw/log storage changed complete generation')
    if raw['episode_memory']['installation']['admission_plan_sha256']!=log['episode_memory']['installation']['admission_plan_sha256']:
        raise ValueError('storage altered logical admission')
    for d,frames in ((raw,source),(log,source),(away,wrong)):
        if d['episode_memory']['capture']['source_frames']!=frames:raise ValueError('wrong causal source episode')
        if not d['episode_memory']['installation']['cache_metadata_unchanged']:raise ValueError('cache positions changed')
    rb=raw['episode_memory']['ledger'];lb=log['episode_memory']['ledger'];ab=away['episode_memory']['ledger']
    if not rb['demand_H2D_payload_bytes']==ab['demand_H2D_payload_bytes']>0:raise ValueError('wrong-memory byte control differs')
    if not 0<lb['CPU_archive_peak_bytes']<rb['CPU_archive_peak_bytes']:raise ValueError('log is not smaller')
    if reports['none']['episode_memory']['ledger']['CPU_archive_peak_bytes']:raise ValueError('baseline archived hidden KV')
    return dict(status='pass',cases=4,source_scenario=raw['cut_scenario'],seed=raw['seed'],gpu=raw['gpu'],
        same_noise_and_pre_return_prefix=True,raw_log_full_latent_RGB_exact=True,same_raw_log_admission_SHA=True,
        raw_wrong_memory_payload_equal=True,CPU_archive_raw_over_log=rb['CPU_archive_peak_bytes']/lb['CPU_archive_peak_bytes'],
        ledgers={mode:d['episode_memory']['ledger'] for mode,d in reports.items()},
        quality='pending_own_generated_information_review',privileged_admission_not_online_method=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gate',action='store_true')
    args=p.parse_args();rows=[]
    groups=[args.root] if args.gate else [args.root/f'lane{i}' for i in range(4)]
    for lane,root in enumerate(groups):
        try:
            reports={m:json.loads((root/m/'summary.json').read_text()) for m in MODES}
            if not args.gate and any((d['cut_scenario'],d['seed'])!=DESIGN[lane] for d in reports.values()):
                raise ValueError('wrong frozen scenario/seed')
            row=validate_group(reports,gate=args.gate)
            row.update(lane=lane,source_summary_sha256={m:hashlib.sha256((root/m/'summary.json').read_bytes()).hexdigest() for m in MODES})
        except (OSError,KeyError,ValueError) as error:
            row=dict(lane=lane,status='fail',error=str(error),partial_artifacts_preserved=True)
        rows.append(row)
    report=dict(status='pass' if all(r['status']=='pass' for r in rows) else 'fail',groups=rows,
        expected_cases=len(groups)*4,completed_contract_cases=sum(r.get('cases',0) for r in rows))
    with (args.root/'episode_admission_storage_audit.json').open('x') as handle:json.dump(report,handle,indent=2);handle.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='groups'}))
    if report['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
