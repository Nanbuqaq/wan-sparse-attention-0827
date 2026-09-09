#!/usr/bin/env python3
"""Offline, fixed-Q sensitivity to historical key phase/virtual-time alignment."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.native_temporal_rephase import rephase_temporal_keys
from scripts.analyze_native_attention_teacher import decompose,output_error


def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    report=json.loads((args.capture/'summary.json').read_text())
    assert report['status']=='pass' and report['observer_noise_latent_RGB_equivalence']
    assert report['episode_memory']['capture']['source_frames']==list(range(40,48))
    path=args.capture/'attention_teacher.pt'
    with path.open('rb') as handle:sha=hashlib.file_digest(handle,'sha256').hexdigest()
    assert sha==report['attention_teacher']['sha256']
    records=torch.load(path,map_location='cpu',weights_only=True)['records'];rows=[]
    for record in records:
        baseline,base=decompose(record,['initial','source','away','current'])
        for name,delta in (('original',0),('current_phase_only',16),('recent_virtual_time_and_phase',64)):
            changed=dict(record)
            if delta:
                changed['k']=record['k'].clone()
                changed['k'][:,7040:14080]=rephase_temporal_keys(record['k'][:,7040:14080],delta)
            analysis,values=decompose(changed,['initial','source','away','current'])
            rows.append(dict(phase=record['phase'],layer=record['layer'],policy=name,delta=delta,
                role_mass={k:v['probability_mass']['mean'] for k,v in analysis['roles'].items()},
                source_lower_center_mass=analysis['roles']['source']['mass_by_query_site']['lower_center']['mean'],
                output_change_from_original=output_error(base['fp32_output'],values['fp32_output']),
                original_native_numeric_gate=baseline['FP32_replay_gate'],
                changed_output_is_not_expected_to_match_original=True))
    result=dict(capture_sha256=sha,rows=rows,scope='fixed captured queries and values; transformed temporal K channels only',
        video_quality_evaluated=False,already_rounded_K_not_original_preRoPE_K=True,
        source_actual_frames=[40,48],current_query_frames=[96,104],
        policies=dict(current_phase_only='source frame40/phase8 to frame40/phase24',
                      recent_virtual_time_and_phase='source frame40/phase8 to virtualframe88/phase24'))
    (args.output/'rephase_sensitivity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
