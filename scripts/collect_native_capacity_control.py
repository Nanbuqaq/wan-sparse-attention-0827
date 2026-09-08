#!/usr/bin/env python3
"""Do not mix new-compiler capacity controls with the queued H200 cohort."""
import argparse
import hashlib
import json
from pathlib import Path


def validate_full_pair(a,b):
    for d,window in ((a,32),(b,128)):
        if (d['status']!='pass' or d['latent_shape']!=[1,128,48,44,80] or d['pixels']['frames']!=509
                or d['local_frames']!=window or d['episode_memory_mode'] is not None
                or d['native_KV_allocation_policy']!='CFG1_positive_only' or d['triton_version']!='3.3.1'
                or not d['torch_dynamo_disabled'] or d['attention_backend']!='native_FA2' or d['fallback_allowed']):
            raise ValueError('wrong native capacity/runtime identity')
        if d['native_positive_and_negative_KV_bytes']!=window*880*24*128*2*2*30:
            raise ValueError('actual allocated KV does not match window')
        if d['fixed_native_adaln_recipe']!=dict(num_warps=16,num_stages=1):raise ValueError('unfrozen recipe')
    for key in ('seed','cut_scenario','gpu','runner_commit','assets_manifest_sha256','noise_sha256'):
        if a[key]!=b[key]:raise ValueError('capacity pair differs in '+key)
    return dict(status='pass',same_noise=True,own_source_trajectory_required=True,
        window32_to_128_KV_ratio=4,quality='pending_absence_and_own_source_review',cross_compiler_equivalence_claimed=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    groups=[];states=[]
    for lane in range(4):
        root=args.root/f'lane{lane}';reports={};rows=[]
        for case in ('gate_native','gate_positive','window32','window128'):
            path=root/case/'summary.json'
            try:
                d=json.loads(path.read_text());reports[case]=d
                row=dict(case=case,status=d['status'],stage=d.get('stage'),summary=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            except (OSError,ValueError) as error:row=dict(case=case,status='missing',error=str(error))
            rows.append(row);states.append(row['status'])
        try:
            gate=json.loads((root/'hardware_gate.json').read_text())
            if gate['status']!='pass':raise ValueError('hardware gate failed')
            plan=json.loads((root/'lane_plan.json').read_text())
            for name,d in reports.items():
                if d['cut_scenario']!=plan['scenario'] or d['seed']!=plan['gate_seed' if name.startswith('gate_') else 'seed']:
                    raise ValueError('case differs from frozen lane identity')
            contract=validate_full_pair(reports['window32'],reports['window128'])
        except (OSError,KeyError,ValueError) as error:contract=dict(status='fail',error=str(error))
        groups.append(dict(lane=lane,contract=contract,cases=rows))
    result=dict(status='pass' if all(g['contract']['status']=='pass' for g in groups) else 'fail',
        groups=groups,expected_executions=16,full_capacity_cases=8,
        technical_pass=states.count('pass'),technical_fail=states.count('fail'),missing=states.count('missing'))
    with (args.root/'capacity_control_terminal.json').open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='groups'}))
    if result['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
