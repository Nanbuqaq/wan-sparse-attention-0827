#!/usr/bin/env python3
"""Same-compiler allocation gate, then distinct full native capacity controls."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def validate_gate(native,positive):
    for d in (native,positive):
        if (d['status']!='pass' or d['latent_shape']!=[1,64,48,16,32] or d['local_frames']!=32
                or d['triton_version']!='3.3.1' or not d['torch_dynamo_disabled']
                or d['attention_backend']!='native_FA2' or d['fallback_allowed']):
            raise ValueError('new hardware gate identity/backend differs')
    for k in ('noise_sha256','latent_sha256','seed','cut_scenario','gpu','runner_commit','assets_manifest_sha256'):
        if native[k]!=positive[k]:raise ValueError('same-compiler gate changed '+k)
    if native['pixels']['raw_RGB_sha256']!=positive['pixels']['raw_RGB_sha256']:
        raise ValueError('allocation changed RGB')
    if (native['native_KV_allocation_policy']!='native_positive_and_negative'
            or positive['native_KV_allocation_policy']!='CFG1_positive_only'
            or native['native_positive_and_negative_KV_bytes']!=2*positive['native_positive_and_negative_KV_bytes']):
        raise ValueError('allocation was not reduced exactly')
    return dict(status='pass',same_compiler_native_positive_latent_RGB_exact=True,
                cross_compiler_equivalence_claimed=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cut-scenario',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--gate-seed',type=int,required=True)
    p.add_argument('--reverse-full-order',action='store_true');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'lane_plan.json').write_text(json.dumps(dict(scenario=args.cut_scenario,seed=args.seed,
        gate_seed=args.gate_seed,windows=[128,32] if args.reverse_full_order else [32,128],
        expected_compiler='3.3.1',cross_compiler_equivalence_claimed=False),indent=2)+'\n')
    def run(name,seed,extra):
        cmd=[sys.executable,str(ROOT/'scripts/run_longlive2_native_reference.py'),
            '--assets',str(args.assets.resolve()),'--output',str((args.output/name).resolve()),
            '--cut-scenario',args.cut_scenario,'--seed',str(seed),'--fixed-adaln-warps','16','--fixed-adaln-stages','1',*extra]
        with (args.output/f'{name}.log').open('x') as log:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
        path=args.output/name/'summary.json'
        if not path.exists():
            path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(dict(status='fail',stage='missing_child_report',exit_code=code),indent=2)+'\n')
        d=json.loads(path.read_text());print(json.dumps(dict(case=name,status=d['status'],exit_code=code)),flush=True);return d
    gate_args=['--gate','--episode-gate-layout','--native-local-frames','32']
    native=run('gate_native',args.gate_seed,gate_args)
    gate=dict(status='fail',reason='native_hardware_gate_failed')
    if native['status']=='pass':
        positive=run('gate_positive',args.gate_seed,[*gate_args,'--cfg1-positive-cache-only',
            '--equivalence-reference',str((args.output/'gate_native/summary.json').resolve())])
        try:gate=validate_gate(native,positive)
        except (KeyError,ValueError) as error:gate=dict(status='fail',reason=str(error))
    else:
        (args.output/'gate_positive').mkdir()
        (args.output/'gate_positive/summary.json').write_text(json.dumps(dict(status='fail',stage='not_started_after_native_hardware_gate',quality_evaluated=False),indent=2)+'\n')
    (args.output/'hardware_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    rows=[]
    for window in ((128,32) if args.reverse_full_order else (32,128)):
        name=f'window{window}'
        if gate['status']=='pass':
            d=run(name,args.seed,['--native-local-frames',str(window),'--cfg1-positive-cache-only'])
        else:
            (args.output/name).mkdir();d=dict(status='fail',stage='not_started_after_same_compiler_hardware_gate',quality_evaluated=False)
            (args.output/name/'summary.json').write_text(json.dumps(d,indent=2)+'\n')
        rows.append(dict(case=name,status=d['status'],stage=d.get('stage')))
    (args.output/'lane_terminal.json').write_text(json.dumps(dict(gate=gate,full_cases=rows,missing=0),indent=2)+'\n')
    if gate['status']!='pass' or any(r['status']!='pass' for r in rows):raise SystemExit(1)


if __name__=='__main__':main()
