#!/usr/bin/env python3
"""Frozen source-block cases; independent GPU lanes and append-only outcomes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def build_cases(spec, stage, assets, source, output):
    methods = spec['methods']
    if stage == 'gate':
        methods = [m for m in methods if m['id'] in spec['gate_methods']]
    scenarios = spec['scenarios'] if stage == 'screen' else spec['scenarios'][1:]
    cases = []
    for scenario in scenarios:
        for method in methods:
            name = scenario + '__' + method['id']
            command = [sys.executable, str(ROOT/'scripts/run_longlive2_native_reference.py'),
                '--assets', str(assets), '--source', str(source), '--output', str(output/name),
                '--cut-scenario', scenario, '--seed', str(spec[stage+'_seed']),
                '--native-local-frames', '32', '--cfg1-positive-cache-only',
                '--fixed-adaln-warps', '16', '--fixed-adaln-stages', '1',
                '--constructor-mode', 'strict_checkpoint_no_parameter_init', '--native-inplace-cache']
            if stage == 'gate':
                command += ['--gate', '--episode-gate-layout']
            if spec.get('shared_conditioning'):
                command += ['--native-shared-conditioning']
            if method['policy']:
                command += ['--causal-block-policy', method['policy'], '--causal-block-fraction', str(method['fraction']),
                    '--causal-block-grouping', method['grouping'], '--causal-block-heads', method['heads']]
                if method.get('query_reduction'):
                    command += ['--causal-block-query-reduction', method['query_reduction']]
            cases.append(dict(id=name, scenario=scenario, method=method, cmd=command))
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/system/native_causal_block_wave1.json')
    parser.add_argument('--source', type=Path, default=ROOT/'third_party/LongLive2')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('gate', 'screen'), required=True)
    parser.add_argument('--required-gpu-name', default='')
    parser.add_argument('--allow-h800', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    spec_path = args.config
    spec = json.loads(spec_path.read_text())
    cases = build_cases(spec, args.stage, args.assets, args.source, args.output)
    visible = [x for x in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if x]
    sha = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    plan = dict(code_sha=sha, config_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                cases=cases, visible_devices=visible, stage=args.stage)
    if not args.run:
        print(json.dumps(plan, indent=2)); return
    if not visible or len(visible) != len(set(visible)):
        raise ValueError('unique assigned GPU devices or inherited physical locks required')
    if len(visible)>len(cases):
        raise ValueError('each reserved lane must have actual GPU work')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'batch_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    topology=[sys.executable,str(ROOT/'scripts/check_native_hardware.py'),'--expected-count',str(len(visible)),
        '--required',args.required_gpu_name,'--output',str(args.output/'hardware.json')]
    if args.allow_h800:topology+=['--allow-h800']
    if subprocess.call(topology):
        (args.output/'batch_terminal.json').write_text(json.dumps(dict(code_sha=sha,status='fail',
            rows=[dict(id=c['id'],status='blocked_by_hardware_topology',returncode=1) for c in cases]))+'\n')
        raise SystemExit(1)
    accepted=(args.required_gpu_name,'H800') if args.allow_h800 else (args.required_gpu_name,)

    def lane(index):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = visible[index]
        for key in ('WAN_SPARSE_PHYSICAL_GPU', 'WAN_SPARSE_PHYSICAL_GPUS'):
            env.pop(key, None)
        env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1',
            PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false', TRANSFORMERS_OFFLINE='1', HF_HUB_OFFLINE='1',
            LLV2_USE_FA3='0', LLV2_USE_FA4='0', LLV2_USE_TE_ATTN='0', LLV2_COMPILE_VAE='0',
            PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        rows = []
        # Check assigned hardware before model work; every case then executes native CUDA.
        check = 'import torch; assert torch.cuda.is_available(); n=torch.cuda.get_device_name(0); print(n); assert any(x in n for x in '+repr(accepted)+')'
        with (args.output/f'lane{index}_hardware.log').open('x') as handle:
            gate = subprocess.call([sys.executable, '-c', check], env=env, stdout=handle, stderr=subprocess.STDOUT)
        gate_kind='hardware'
        if not gate:
            gate_kind='component'
            with (args.output/f'lane{index}_component.log').open('x') as handle:
                gate=subprocess.call([sys.executable,str(ROOT/'scripts/gate_native_resident_component.py'),
                    '--output',str(args.output/f'component_lane{index}')],env=env,stdout=handle,stderr=subprocess.STDOUT)
        for case in cases[index::len(visible)]:
            row = dict(id=case['id'], lane=index, scenario=case['scenario'], method=case['method']['id'])
            if gate:
                row.update(status='blocked_by_'+gate_kind+'_gate', returncode=gate)
            else:
                try:
                    log = args.output/(case['id']+'.log')
                    with log.open('x') as handle:
                        code = subprocess.call(case['cmd'], env=env, stdout=handle, stderr=subprocess.STDOUT)
                    path = args.output/case['id']/'summary.json'
                    data = json.loads(path.read_text()) if path.exists() else {}
                    status = data.get('status', 'missing')
                    if code or not any(x in data.get('gpu','') for x in accepted):
                        status = 'fail'
                    row.update(returncode=code, status=status, summary=str(path), log=str(log))
                except Exception as error:
                    row.update(returncode=-1, status='fail', error=repr(error))
            rows.append(row)
            with (args.output/f'lane{index}_progress.jsonl').open('a') as handle:
                handle.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)
        return rows

    with ThreadPoolExecutor(max_workers=len(visible)) as pool:
        rows = [row for result in pool.map(lane, range(len(visible))) for row in result]
    terminal = dict(code_sha=sha, rows=rows, stage=args.stage, quality_review_complete=False,
        status='pass' if len(rows)==len(cases) and all(r['status']=='pass' and r['returncode']==0 for r in rows) else 'fail')
    (args.output/'batch_terminal.json').write_text(json.dumps(terminal, indent=2)+'\n')
    if terminal['status'] != 'pass':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
