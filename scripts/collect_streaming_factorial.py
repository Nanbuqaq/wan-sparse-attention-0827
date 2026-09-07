#!/usr/bin/env python3
"""Terminal and paired 2x2 audit of complete streaming/RoPE repetitions."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize_lane(summary, repeats):
    arms=['batch_current_stream__rope_upstream','batch_current_stream__rope_direct_output',
          'async_priority_current_stream__rope_upstream','async_priority_current_stream__rope_direct_output']
    expected={(a,r) for a in arms for r in range(repeats)}
    records=summary['variants']
    if len(records)!=len(expected) or {(r['variant'],r['repetition']) for r in records}!=expected:
        raise ValueError('missing, extra, or duplicate factorial case')
    if any(r['status']!='pass' or not r['exact_reference'] or r['diagnostic_timing'] for r in records):
        raise ValueError('non-equivalent or profiled trajectory in timing matrix')
    if len({json.dumps(r['identity'],sort_keys=True) for r in records})!=1:
        raise ValueError('full trajectories are not exactly equal')
    rows={}
    for arm in arms:
        rs=sorted([r for r in records if r['variant']==arm],key=lambda r:r['repetition'])
        times=[r['generation_decode_encode_s'] for r in rs]
        rows[arm]=dict(complete_service_s=times,median_s=statistics.median(times),minimum_s=min(times),maximum_s=max(times),
            first_packet_muxed_s=[r['sink']['first_packet_muxed_s'] for r in rs],
            peak_GPU_bytes=[r['peak_GPU_bytes'] for r in rs])
    contrast={}
    for name,left,right in [('RoPE_under_batch',arms[0],arms[1]),('RoPE_under_streaming',arms[2],arms[3]),
                            ('streaming_under_upstream_RoPE',arms[0],arms[2]),('streaming_under_direct_RoPE',arms[1],arms[3]),
                            ('combined',arms[0],arms[3])]:
        ratios=[x/y for x,y in zip(rows[left]['complete_service_s'],rows[right]['complete_service_s'])]
        contrast[name]=dict(paired_speedups=ratios,median=statistics.median(ratios),minimum=min(ratios),maximum=max(ratios))
    return dict(status='pass',method=summary['method'],prompt=summary['prompt'],gpu=summary['gpu'],
        seed=summary['seed'],latent_frames=summary['latent_frames'],source_commit=summary['source_commit'],
        cases=len(records),same_noise_routes_latents_RGB=True,arms=rows,contrasts=contrast,
        samples_are_same_process_blocked_repetitions=True,independent_machine_replications=False,
        service_scope='generation + VAE + incremental encode and flush; excludes load and post-hoc artifact audit')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--expected-lanes',type=int,required=True);p.add_argument('--repeats',type=int,required=True)
    p.add_argument('--exit-codes',type=int,nargs='+',required=True)
    args=p.parse_args()
    if len(args.exit_codes)!=args.expected_lanes: raise ValueError('exit-code/lane mismatch')
    lanes=[]
    for lane,code in enumerate(args.exit_codes):
        path=args.root/f'lane{lane}'/'summary.json'
        try:
            if code!=0: raise ValueError(f'lane exit code {code}')
            summary=json.loads(path.read_text())
            if summary['status']!='pass': raise ValueError('lane not successful')
            row=summarize_lane(summary,args.repeats)
            row['summary_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            row['lane']=lane
        except (ValueError,KeyError,OSError) as exc:
            row=dict(lane=lane,status='fail',exit_code=code,error=str(exc),partial_artifacts_preserved=True)
        lanes.append(row)
    result=dict(status='pass' if all(r['status']=='pass' for r in lanes) else 'fail',lanes=lanes,
        expected_cases=args.expected_lanes*args.repeats*4,completed_cases=sum(r.get('cases',0) for r in lanes),
        failed_lanes=sum(r['status']!='pass' for r in lanes),different_GPU_models_not_pooled=True)
    with (args.root/'factorial_audit.json').open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='lanes'}))
    if result['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
