#!/usr/bin/env python3
"""Delivery cadence and representative quality boards for exact 957 pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_streaming_length_extension import summarize
from scripts.build_video_review_storyboards import analyze


def delivery_metrics(records,*,start_latent=30,fps=16):
    if len(records)<2:raise ValueError('multiple complete chunks required')
    gaps=[];budgets=[]
    for previous,current in zip(records,records[1:]):
        if current['start_latent']<start_latent:continue
        gap=current['sink_completed_s']-previous['sink_completed_s']
        if gap<=0:raise ValueError('nonmonotone completed sink times')
        gaps.append(gap);budgets.append(current['pixel_frames']/fps)
    if not gaps:raise ValueError('empty steady window')
    return dict(chunks=len(gaps),start_latent=start_latent,delivery_gap_p50_s=float(np.median(gaps)),
        delivery_gap_p95_s=float(np.percentile(gaps,95)),delivery_gap_max_s=max(gaps),
        target_fps=fps,per_chunk_playback_budget_s=sorted(set(budgets)),
        fraction_meeting_per_chunk_playback_budget=float(np.mean(np.array(gaps)<=budgets)),
        scope='server_sink_completion_cadence_not_client_display_or_GPU_service')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);groups=[]
    for name in ('calibration_motion','calibration_state'):
        source=args.root/name/'summary.json';report=json.loads(source.read_text());verified=summarize(report)
        deliveries=[]
        for row in report['variants']:
            decoder=row.get('streaming_decoder')
            if decoder:
                deliveries.append(dict(repetition=row['repetition'],steady=delivery_metrics(decoder['records']),
                    late_quarter=delivery_metrics(decoder['records'],start_latent=180),
                    producer_backpressure_s=decoder['producer_backpressure_s'],
                    pinned_output_bytes=decoder['pinned_output_bytes'],
                    CPU_outputs_collected=decoder['collected_CPU_outputs_unbounded_by_slot_budget']))
        primary=next(r for r in report['variants'] if r['variant']=='batch_current_stream' and r['repetition']==0)
        video=args.root/name/(primary['variant']+'__rep00')/'video.mp4'
        boards=analyze(dict(id=name,video=str(video),latent_frames=240,status='pass'),args.output,samples_per_quarter=16)
        groups.append(dict(prompt=name,gpu=report['gpu'],source_commit=report['source_commit'],
            source_summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            exact_repeats_and_arms=True,paired_speedups=verified['paired_speedups'],
            median_paired_speedup=verified['median_paired_speedup'],delivery=deliveries,
            one_representative_for_six_exact_RGB_trajectories=boards,
            absolute_quality='pending_descriptive_review'))
    (args.output/'summary.json').write_text(json.dumps(dict(status='pass',groups=groups,
        first_packet_is_not_first_client_display=True,independent_trials_not_claimed=True),indent=2)+'\n')
    print(json.dumps(dict(status='pass',groups=2,executions=12,unique_RGB_trajectories=2)))


if __name__=='__main__':main()
