#!/usr/bin/env python3
"""Frozen72-point dataflow characterization: resident, onload and Page256 stream.

Synthetic head-major CPU archive; not an end-to-end LongLive speed benchmark.
Routes/density/reuse are fixed geometrically before any FP32 teacher evaluation.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.dataflow_reference import DataflowInputs, PreparedDataflow, StreamingDataflow
from adapters.longlive_sparse.offline_eval import output_error_metrics
from scripts.gate_dataflow_references import teacher, PreparedFA2, allowed_keys


def cases():
    return [{'index': i, 'query_tokens': q, 'history_frames': frames, 'density': density,
             'reuse': reuse, 'layout': layout, 'seed': 2026090700+i}
        for i, (q, frames, density, reuse, layout) in enumerate(itertools.product(
            (1560, 4680), (1, 6, 12), (.1, .25), (1, 2, 3), ('contiguous', 'fragmented')))]


def selection(case):
    frames, used = case['history_frames'], round(case['history_frames']*1560*case['density'])
    if case['layout'] == 'contiguous':
        indices = np.arange(used, dtype=np.int64)
    else:
        blocks = [(f, b, min(b+64, 1560)) for f in range(frames) for b in range(0, 1560, 64)]
        shuffled = np.random.default_rng(case['seed']).permutation(len(blocks)).tolist()
        # Ensure all three query groups have at least one key at the shortest shape.
        first = [next(i for i in shuffled if i%3 == role) for role in range(3)]
        order = first+[i for i in shuffled if i not in first]
        indices = np.concatenate([f*1560+np.arange(start, end) for f, start, end in (blocks[i] for i in order)])[:used]
    role = ((indices//1560)*25+(indices%1560)//64)%3
    return torch.from_numpy(indices), torch.from_numpy(role.astype(np.int32))


def errors_pass(error):
    return error['max_abs'] <= .02 and error['relative_l2'] <= .01 and error['one_minus_cosine'] <= .001


def sample(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    rows = []
    for _ in range(repeats):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        begin.record()
        fn()
        end.record()
        end.synchronize()
        rows.append({'wall_ms': (time.perf_counter()-start)*1000, 'gpu_span_ms': begin.elapsed_time(end)})
    return {'samples': rows,
        **{f'{kind}_{stat}': float(np.percentile([r[kind] for r in rows], percentile))
           for kind in ('wall_ms', 'gpu_span_ms') for stat, percentile in (('median', 50), ('p95', 95))}}


def event_spans(stream):
    origin, records = stream.last_events
    copies = [(origin.elapsed_time(a), origin.elapsed_time(b)) for a, b, _, _ in records]
    kernels = [(origin.elapsed_time(a), origin.elapsed_time(b)) for _, _, a, b in records]
    intersection = sum(max(0., min(b, d)-max(a, c)) for a, b in copies for c, d in kernels)
    return {'copy_spans_ms': copies, 'compute_spans_ms': kernels,
        'span_intersection_ms_not_actual_overlap_proof': intersection, 'requires_Nsight_activity_audit': True}


@torch.inference_mode()
def benchmark(case, args):
    h, d = 12, 128
    q, full_k = case['query_tokens'], case['history_frames']*1560
    indices, roles = selection(case)
    k = len(indices)
    torch.manual_seed(case['seed'])
    setup = time.perf_counter()
    cpu_k = torch.randn(h, full_k, d, device='cuda', dtype=torch.bfloat16).cpu()
    cpu_v = torch.randn(h, full_k, d, device='cuda', dtype=torch.bfloat16).cpu()
    c = DataflowInputs(torch.randn(h, q, d, device='cuda', dtype=torch.bfloat16),
        cpu_k.index_select(1, indices).cuda(), cpu_v.index_select(1, indices).cuda(), roles.cuda(), case['reuse'])
    torch.cuda.synchronize()
    input_setup_s = time.perf_counter()-setup
    # Frozen geometry, not teacher-driven admission.
    route_sha = hashlib.sha256(indices.numpy().tobytes()+roles.numpy().tobytes()+json.dumps(case, sort_keys=True).encode()).hexdigest()
    target = teacher(c)
    prepared = PreparedDataflow(c)
    allocate_started = time.perf_counter()
    prepared.allocate_kv_workspace()
    torch.cuda.synchronize()
    workspace_alloc_s = time.perf_counter()-allocate_started
    prep_started = time.perf_counter()
    fa2 = PreparedFA2(c)
    torch.cuda.synchronize()
    fa2_prepare_s = time.perf_counter()-prep_started
    host_k = torch.empty_like(c.key, device='cpu', pin_memory=True)
    host_v = torch.empty_like(host_k, pin_memory=True)
    stream = StreamingDataflow(c, cpu_k, cpu_v, indices)
    methods = {'qout_gpu_reference': lambda: prepared.qout()[0],
               'kv_stationary_split_reference': lambda: prepared.kvout()[0], 'grouped_fa2': fa2}
    rows = []
    for name, resident in methods.items():
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = resident()
        torch.cuda.synchronize()
        first = time.perf_counter()-start
        error = output_error_metrics(target, result)
        if not errors_pass(error):
            raise ValueError(f'{name} numerical failure: {error}')
        for scope in ('resident', 'onload_inclusive'):
            def invoke():
                if scope == 'onload_inclusive':
                    torch.index_select(cpu_k, 1, indices, out=host_k)
                    torch.index_select(cpu_v, 1, indices, out=host_v)
                    c.key.copy_(host_k, non_blocking=True)
                    c.value.copy_(host_v, non_blocking=True)
                    if name == 'grouped_fa2':
                        fa2.refresh_kv(c)
                return resident()
            timing = sample(invoke, args.warmup, args.repeats)
            checked_output = invoke()
            torch.cuda.synchronize()
            scope_error = output_error_metrics(target, checked_output)
            if not errors_pass(scope_error):
                raise ValueError(f'{name}/{scope} numerical failure: {scope_error}')
            rows.append({'backend': name, 'scope': scope, 'first_resident_call_including_compile_s': first,
                'bf16_vs_fp32': scope_error, **timing, 'kv_H2D_bytes_per_call': 0 if scope == 'resident' else 2*h*k*d*2,
                'KV_padding_bytes': 0, 'actual_overlap_proven': False})
    for backend in ('qout', 'kvout'):
        for overlap in (False, True):
            scope = 'streaming_overlap' if overlap else 'streaming_serial'
            def invoke():
                return stream.run(backend, overlap=overlap)
            start = time.perf_counter()
            result = invoke()
            torch.cuda.synchronize()
            first = time.perf_counter()-start
            error = output_error_metrics(target, result)
            if not errors_pass(error):
                raise ValueError(f'{backend}/{scope} numerical failure: {error}')
            if args.profile_scope == scope and args.profile_backend == backend:
                for _ in range(args.warmup):
                    invoke()
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
                torch.cuda.nvtx.range_push(f'dataflow_case{case["index"]}_{backend}_{scope}')
                invoke()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
                torch.cuda.cudart().cudaProfilerStop()
            timing = sample(invoke, args.warmup, args.repeats)
            stream.run(backend, overlap=overlap, record_events=True)
            torch.cuda.synchronize()
            rows.append({'backend': backend+'_gpu_page_reference', 'scope': scope,
                'first_streaming_call_including_compile_s': first, 'bf16_vs_fp32': error, **timing,
                'kv_H2D_bytes_per_call': 2*h*k*d*2, 'KV_padding_bytes': 0,
                'copy_api_calls': 2*((k+255)//256), 'actual_DMA_copy_count': None,
                'event_span_audit': event_spans(stream), 'actual_overlap_proven': False})
    qgroups = [int(((torch.arange(q)*3//q) == group).sum()) for group in range(3)]
    key_counts = [int(allowed_keys(roles, group, case['reuse']).sum()) for group in range(3)]
    logical = h*sum(a*b for a, b in zip(qgroups, key_counts))
    return {'status': 'pass', 'case': case, 'records': rows, 'route_sha256': route_sha,
        'gpu': torch.cuda.get_device_name(), 'compute_capability': torch.cuda.get_device_capability(),
        'torch_version': torch.__version__, 'heads': h, 'head_dim': d, 'dtype': 'bfloat16',
        'triton_cache_dir': os.environ.get('TRITON_CACHE_DIR'),
        'warmup': args.warmup, 'repeats': args.repeats, 'input_setup_s': input_setup_s,
        'kv_workspace_alloc_s': workspace_alloc_s, 'kv_partial_workspace_bytes': prepared.workspace_bytes,
        'qout_streaming_state_bytes': 4*h*q*(d+2), 'host_streaming_staging_bytes': stream.staging_bytes,
        'FA2_group_prepare_s': fa2_prepare_s, 'FA2_group_prepared_bytes': fa2.prepared_bytes,
        'actual_union_tokens_per_head': k, 'physical_history_density': k/full_k,
        'logical_pairs': logical, 'logical_history_density': logical/(h*q*full_k),
        'Q_group_sizes': qgroups, 'group_key_counts': key_counts,
        'scope': 'synthetic_history_only_stationary_GPU_references_not_video_or_tuned_KVOut',
        'CPU_archive_layout': 'synthetic_head_major; initial representation setup reported separately',
        'no_full_candidate_pin_omitted_from_onload': True,
        'FA2_internal_MMA_padding_not_measured': True,
        'online_video_promotion': False, 'fallback_calls': 0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--lane', type=int, default=0)
    p.add_argument('--lanes', type=int, default=1)
    p.add_argument('--case-index', type=int)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeats', type=int, default=30)
    p.add_argument('--profile-scope', choices=('streaming_serial', 'streaming_overlap'))
    p.add_argument('--profile-backend', choices=('qout', 'kvout'), default='kvout')
    args = p.parse_args()
    if args.warmup < 1 or args.repeats < 1 or not 0 <= args.lane < args.lanes:
        p.error('invalid measurement or lane counts')
    if args.profile_scope and args.case_index is None:
        p.error('profile only a single representative case')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('real GPU matrix required')
    selected = [c for c in cases() if (c['index'] == args.case_index if args.case_index is not None else c['index']%args.lanes == args.lane)]
    if not selected:
        p.error('empty case selection')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    (root/'manifest.json').write_text(json.dumps({'source': source, 'cases': selected, 'args': vars(args)}, indent=2)+'\n')
    states = []
    for case in selected:
        try:
            result = benchmark(case, args)
        except Exception as error:
            result = {'status': 'fail', 'case': case, 'failure_reason': repr(error), 'traceback': traceback.format_exc()}
        result['source_commit'] = source
        path = root/f'case{case["index"]:02d}.json'
        path.write_text(json.dumps(result, indent=2)+'\n')
        states.append({'case': case['index'], 'status': result['status'], 'path': str(path),
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        print(json.dumps(states[-1]), flush=True)
        torch.cuda.empty_cache()
    (root/'terminal.json').write_text(json.dumps({'status': 'pass' if all(s['status'] == 'pass' for s in states) else 'fail',
        'cases': states, 'missing': 0}, indent=2)+'\n')
    if any(s['status'] != 'pass' for s in states):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
