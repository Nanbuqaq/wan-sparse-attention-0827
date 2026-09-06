#!/usr/bin/env python3
"""Frozen same-route timing repeats; independent physical GPU/CPU lanes.

Generator checkouts are immutable. Replicate identity is separate from the
unchanged semantic case identity. Failures do not prevent subsequent repeats.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

WORKSPACE = Path('/home/zhouhe08/MyProjects/0904-longlive-system')
SOURCES = {
    'old': ('/tmp/longlive-bootstrap-2ad4465', '2ad4465dc911ac5ad36a81342a3a1bea5cfec479'),
    'new': ('/tmp/longlive-shared-compiler-21393f3', '21393f3d0ad1f13ab6d7c76a36ed883626b10464'),
}
LANES = {
    0: {'prompt': 'motion', 'cpu_ids': list(range(16, 24)), 'order': ['old', 'new', 'new', 'old']},
    1: {'prompt': 'state', 'cpu_ids': list(range(112, 120)), 'order': ['new', 'old', 'old', 'new']},
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def source_gate():
    for path, expected in SOURCES.values():
        observed = subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', path, 'status', '--porcelain'], text=True).strip()
        if observed != expected or dirty:
            raise ValueError(f'generator source not frozen: {path}')
    gates = list((WORKSPACE/'results/metrics/compiler_gpu_gates_21393f3').glob('gpu*.json'))
    if len(gates) != 2 or any(json.loads(p.read_text())['status'] != 'pass' for p in gates):
        raise ValueError('both existing real GPU branch gates must pass')


def prepare(root):
    source_gate()
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for lane, spec in LANES.items():
        for repeat, variant in enumerate(spec['order']):
            run = root/f'lane{lane}'/f'run{repeat}_{variant}'
            repo, commit = SOURCES[variant]
            control = run/'control'
            subprocess.run(['/usr/bin/python3', f'{repo}/scripts/build_dense_system_validation.py',
                '--method', 'transfer_vaware_hybrid_history', '--latent-frames', '120',
                '--prompt-id', f"calibration_{spec['prompt']}", '--lanes', '3',
                '--archive-offload', 'pooled_pageable', '--host-pinned-budget-mib', '128',
                '--output-dir', str(control)], check=True, cwd=repo)
            suite = control/'lane3.json'
            rows.append({'lane': lane, 'repeat': repeat, 'variant': variant, 'source_commit': commit,
                'repo': repo, 'prompt': spec['prompt'], 'cpu_ids': spec['cpu_ids'],
                'run': str(run), 'suite': str(suite), 'suite_sha256': digest(suite)})
    manifest = {'schema': 1, 'status': 'frozen', 'cases': rows, 'expected_runs': 8,
        'scope': 'same_seed_same_route_repeated_timing_not_independent_quality_samples',
        'generator_runtime_gates_reused': 'compiler_gpu_gates_21393f3',
        'cpu_policy': 'taskset 8 physical cores on GPU-local NUMA node; default memory policy',
        'orders': {'motion': 'ABBA', 'state': 'BAAB'},
        'primary': 'per-prompt median complete generation/artifact latency; model load separate',
        'secondary': 'routing, gather, H2D, materialize, encode/load timings and identical bytes',
        'stop': 'no efficacy promotion on failed route/latent/video equality',
        'previous_unbound_pairs_not_pooled': True}
    write_new(root/'manifest.json', manifest)
    print(json.dumps({'manifest': str(root/'manifest.json'), 'sha256': digest(root/'manifest.json')}))


def run_lane(root, lane):
    source_gate()
    manifest_path = root/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    rows = [r for r in manifest['cases'] if r['lane'] == lane]
    if len(rows) != 4:
        raise ValueError('exactly four frozen repeats expected per lane')
    lane_root = root/f'lane{lane}'
    write_new(lane_root/'started.json', {'manifest_sha256': digest(manifest_path), 'start_unix': time.time()})
    bundle = Path('/kaimm-distill/zhouhe08/longlive/input_bundle')
    base = WORKSPACE/'publish_repo/third_party/longlive-inferhub'
    rag = WORKSPACE/'publish_repo/third_party/LongLive-RAG'
    terminal = []
    for row in rows:
        run = Path(row['run'])
        if digest(row['suite']) != row['suite_sha256']:
            raise ValueError('frozen suite changed')
        env = {**os.environ, 'LONGLIVE_INPUT_BUNDLE_ROOT': str(bundle),
            'LONGLIVE_BASE_SOURCE': str(base), 'LONGLIVE_RAG_SOURCE': str(rag),
            'LONGLIVE_PYTHON_OVERLAY': str(bundle/'python-overlay'),
            'LONGLIVE_WAN_MODELS_ROOT': str(bundle/'model'),
            'LONGLIVE_GENERATOR_CKPT': str(bundle/'checkpoints/longlive_init.pt'),
            'LONGLIVE_LORA_CKPT': str(bundle/'checkpoints/longlive_lora_003000.pt'),
            'LONGLIVE_DISABLE_FA3': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_OFFLINE': '1',
            'TOKENIZERS_PARALLELISM': 'false', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
            'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2',
            'MKL_NUM_THREADS': '2', 'LONGLIVE_CAPTURE_COMPLETE_ATTENTION': '0', 'LONGLIVE_NVTX': '0',
            'INFER_OUTPUT_DIR': str(run),
            'PYTHONPATH': f"{bundle/'python-overlay'}:{row['repo']}:{base}:{rag}"}
        cmd = ['/usr/bin/python3', f"{row['repo']}/scripts/run_on_free_gpu.py", '--physical-gpu', str(lane),
            '--', 'taskset', '-c', ','.join(map(str, row['cpu_ids'])), '/usr/bin/python3',
            f"{row['repo']}/scripts/run_loaded_method_suite.py", '--suite', row['suite'],
            '--shard-axis', 'case', '--shard-index', '0', '--shard-count', '1']
        start = time.time()
        print(f"START lane={lane} repeat={row['repeat']} variant={row['variant']}", flush=True)
        with (run/'runner.log').open('x') as log:
            code = subprocess.run(cmd, cwd=row['repo'], env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        record = {**row, 'exit_code': code, 'process_wall_s': time.time()-start,
                  'start_unix': start, 'status': 'pass' if code == 0 else 'fail'}
        write_new(run/'execution.json', record)
        terminal.append(record)
        print(f"END lane={lane} repeat={row['repeat']} exit={code}", flush=True)
    write_new(lane_root/'terminal.json', {'status': 'pass' if all(r['exit_code'] == 0 for r in terminal) else 'fail',
        'cases': terminal, 'missing': 0})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--run-lane', type=int, choices=(0, 1))
    args = parser.parse_args()
    root = Path(args.output).resolve()
    prepare(root) if args.prepare else run_lane(root, args.run_lane)


if __name__ == '__main__':
    main()
