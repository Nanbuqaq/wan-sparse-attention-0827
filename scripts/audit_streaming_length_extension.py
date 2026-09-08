#!/usr/bin/env python3
"""Paired terminal audit for the new two-arm 957 development extension."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(report,repeats=3):
    arms=('batch_current_stream','async_priority_current_stream')
    expected={(a,r) for a in arms for r in range(repeats)}
    rows=report['variants']
    if report['status']!='pass' or report['latent_frames']!=240 or len(rows)!=len(expected):
        raise ValueError('incomplete 957 extension')
    if {(r['variant'],r['repetition']) for r in rows}!=expected:raise ValueError('wrong or duplicate arm/repetition')
    if any(r['status']!='pass' or not r['exact_reference'] or r['diagnostic_timing'] or r['decoded_frames']!=957 for r in rows):
        raise ValueError('non-equivalent/profiled/truncated video')
    if len({json.dumps(r['identity'],sort_keys=True) for r in rows})!=1:raise ValueError('complete identities differ')
    per_arm={}
    for arm in arms:
        selected=sorted((r for r in rows if r['variant']==arm),key=lambda r:r['repetition'])
        per_arm[arm]=dict(complete_service_s=[r['generation_decode_encode_s'] for r in selected],
            first_packet_muxed_s=[r['sink']['first_packet_muxed_s'] for r in selected],
            GPU_peak_bytes=[r['peak_GPU_bytes'] for r in selected],
            archive_storage=[r['archive_storage'] for r in selected])
    ratios=[a/b for a,b in zip(per_arm[arms[0]]['complete_service_s'],per_arm[arms[1]]['complete_service_s'])]
    return dict(status='pass',cases=len(rows),prompt=report['prompt'],method=report['method'],gpu=report['gpu'],
        source_commit=report['source_commit'],same_noise_routes_latents_RGB=True,paired_speedups=ratios,
        median_paired_speedup=statistics.median(ratios),negative_repetitions=sum(v<1 for v in ratios),arms=per_arm,
        samples_are_same_process_blocked_repeats=True,not_independent_machine_repeats=True,
        service_scope='generation_VAE_incremental_encode_flush_excluding_load_and_artifact_audit')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();rows=[]
    for name in ('calibration_motion','calibration_state'):
        path=args.root/name/'summary.json'
        try:
            row=summarize(json.loads(path.read_text()));row['source_summary_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError,KeyError,ValueError) as error:
            row=dict(status='fail',prompt=name,error=str(error),partial_artifacts_preserved=True)
        rows.append(row)
    report=dict(status='pass' if all(r['status']=='pass' for r in rows) else 'fail',groups=rows,
        expected_cases=12,terminal_audited_cases=sum(r.get('cases',0) for r in rows),
        absolute_times_across_hardware_not_pooled=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:json.dump(report,handle,indent=2);handle.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='groups'}))
    if report['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
