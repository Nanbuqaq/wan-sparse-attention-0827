#!/usr/bin/env python3
"""Audit matched controls and record technical pass separately from promotion."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--video-root',required=True)
    parser.add_argument('--quality-root',required=True)
    parser.add_argument('--replay-root',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root=Path(args.video_root).resolve()
    terminal=read(root/'terminal_audit.json')
    if terminal['status']!='pass' or terminal['pass_cases']!=6:
        raise ValueError('six technically passing matched cases required')
    cases=read(root/'states.json')['cases']
    if len({c['initial_noise_sha256'] for c in cases})!=1:
        raise ValueError('initial noise differs')
    methods={'rag_dense','transfer_vaware_hybrid_history','rope_aligned_final_history'}
    groups=[]
    for kind,lane in (('motion',0),('state',1)):
        selected=[c for c in cases if c['prompt_id']==f'calibration_{kind}']
        if {c['method'] for c in selected}!=methods or len(selected)!=3:
            raise ValueError('incomplete method triplet')
        log=root/f'lane{lane}/runner.log'
        text=log.read_text()
        physical=re.search(r'\[gpu\] physical=(\d+)',text)
        runtime_line=next(line[len('RUNTIME '):] for line in text.splitlines() if line.startswith('RUNTIME '))
        hardware=json.loads(runtime_line)
        if physical is None or int(physical.group(1))!=lane:
            raise ValueError('physical lane lock not proven in runner log')
        for case in selected:
            if Path(case['video']).resolve().parent.parent!=root/f'lane{lane}':
                raise ValueError('method video came from another lane/hardware group')
        sparse=[c for c in selected if c['method']!='rag_dense']
        for key in ('transferred_bytes','selected_history_tokens','candidate_history_tokens'):
            if sparse[0][key]!=sparse[1][key]:
                raise ValueError(f'sparse budgets differ: {key}')
        quality=read(Path(args.quality_root)/f'{kind}.json')
        by_method={r['method']:r for r in quality['rows']}
        old,new=by_method['transfer_vaware_hybrid_history'],by_method['rope_aligned_final_history']
        tests={'full_lpips':new['lpips_mean']<=old['lpips_mean'],
               'late_quarter_lpips':new['late_quarter_lpips_mean']<=old['late_quarter_lpips_mean'],
               'latent_relative_l2':new['latent_error']['relative_l2']<=old['latent_error']['relative_l2']}
        groups.append({'prompt':kind,'physical_gpu':lane,'runtime_gpu_name':hardware['gpu_name'],
            'runtime_compute_capability':hardware['compute_capability'],
            'runner_log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),
            'same_process_same_gpu_triplet':True,'same_actual_sparse_budget':True,
            'non_regression_checks':tests,'gate_pass':all(tests.values()),
            'full_lpips_legacy':old['lpips_mean'],'full_lpips_aligned':new['lpips_mean'],
            'late_lpips_legacy':old['late_quarter_lpips_mean'],'late_lpips_aligned':new['late_quarter_lpips_mean']})
    replay=[]
    for kind in ('motion','state'):
        for layer in (0,19):
            for start in (46800,177840):
                path=Path(args.replay_root)/kind/f'layer{layer}_start{start}.json'
                data=read(path)
                if data['status']!='pass' or len(data['records'])!=5:
                    raise ValueError('incomplete denoising replay')
                for row in data['records']:
                    reference=row['records']['executed_per_chunk']['vs_full_teacher']['relative_l2']
                    candidate=row['records']['aligned_first_runtime']['vs_full_teacher']['relative_l2']
                    replay.append({'prompt':kind,'layer':layer,'start':start,'call':row['call'],
                                   'relative_error_ratio':candidate/reference})
    result={'status':'pass','technical_cases':6,'missing_cases':0,'formal_promotion':False,
            'research_outcome':'mixed_not_promoted' if not all(g['gate_pass'] for g in groups) else 'development_gate_only',
            'groups':groups,'sampled_replay_calls':len(replay),
            'sampled_replay_improved':sum(r['relative_error_ratio']<1 for r in replay),
            'sampled_replay_records':replay,
            'replay_trajectory_latents':120,'quality_video_latents':39,
            'same_seed_is_not_same_rng_schedule_across_durations':True,
            'unresolved':['unsampled layers','startup history','duration/RNG-schedule difference',
                          'on-policy trajectory shift','seed/order-only variability'],
            'entire_plan_complete':False}
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as handle:
        json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='sampled_replay_records'},indent=2))


if __name__=='__main__':
    main()
