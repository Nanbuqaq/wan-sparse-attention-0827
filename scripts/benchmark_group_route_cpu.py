#!/usr/bin/env python3
"""Exact old/new grouping admission comparison on CPU-reconstructed real captures."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import probe_group_relations as probe
from adapters.longlive_sparse.group_relations import build_group_relation_route


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    reference_sha = subprocess.check_output(['git', '-C', str(args.reference_root), 'rev-parse', 'HEAD'], text=True).strip()
    if reference_sha != 'e9b601153b46282ea61e10395163a8a8d185cbe6':
        raise ValueError('reference must be the frozen first group-video implementation')
    reference_file = args.reference_root/'adapters/longlive_sparse/group_relations.py'
    spec = importlib.util.spec_from_file_location('adapters.longlive_sparse.group_relations_reference', reference_file)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    audit = json.loads((args.workspace/'results/metrics/matched_trajectory_capture_be00491/trajectory_audit.json').read_text())
    result = dict(status='running', scope='CPU_complete_selector_and_route_digest_not_video_speedup',
        reference_commit=reference_sha, reference_file_sha256=hashlib.sha256(reference_file.read_bytes()).hexdigest(),
        new_file_sha256=hashlib.sha256((ROOT/'adapters/longlive_sparse/group_relations.py').read_bytes()).hexdigest(),
        prototypes_and_Q_summaries_reconstructed_on_CPU=True, warmup=5, repeats=30, records=[])
    for case in audit['cases']:
        if case['method'] != 'rag_dense':
            continue
        for layer in (0, 29):
            path = Path(case['capture_dir'])/f'layer{layer:02d}_start00046800_pass00.pt'
            capture = torch.load(path, map_location='cpu', weights_only=True)
            inputs = []
            def save_inputs(*values, **options):
                if options['density'] == .25:
                    inputs.append((values, dict(options)))
                return build_group_relation_route(*values, **options)
            original = probe.build_group_relation_route
            probe.build_group_relation_route = save_inputs
            try:
                probe.build_routes(capture, 'cpu')
            finally:
                probe.build_group_relation_route = original
            for values, options in inputs:
                samples = {'token_reference': [], 'block_expand': []}
                expected = reference.build_group_relation_route(*values, **options).digest()
                for repeat in range(35):
                    modes = ('token_reference', 'block_expand') if repeat % 2 == 0 else ('block_expand', 'token_reference')
                    for mode in modes:
                        function = reference.build_group_relation_route if mode == 'token_reference' else build_group_relation_route
                        begin = time.perf_counter()
                        route = function(*values, **options)
                        digest = route.digest()
                        elapsed = time.perf_counter()-begin
                        if digest != expected:
                            raise RuntimeError(f'selection changed: {case["prompt"], layer, options}')
                        if repeat >= 5:
                            samples[mode].append(elapsed)
                row = dict(prompt=case['prompt'], layer=layer, grouping=options['grouping'], admission=options['admission'],
                    same_route_SHA=True, route_sha256=expected, samples_s=samples,
                    median_s={k: statistics.median(v) for k, v in samples.items()},
                    p95_s={k: float(np.percentile(v, 95)) for k, v in samples.items()})
                row['speedup'] = row['median_s']['token_reference']/row['median_s']['block_expand']
                result['records'].append(row)
                print(json.dumps({k: v for k, v in row.items() if k != 'samples_s'}), flush=True)
            del capture, inputs
    result['status'] = 'pass'
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
