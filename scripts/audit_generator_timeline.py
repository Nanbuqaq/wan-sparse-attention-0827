#!/usr/bin/env python3
"""Measured CPU NVTX scopes and GPU activity unions; never sum nested spans."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3


def merge(intervals):
    output = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if output and start <= output[-1][1]:
            output[-1] = (output[-1][0], max(output[-1][1], end))
        else:
            output.append((start, end))
    return output


def duration(intervals):
    return sum(b-a for a, b in merge(intervals))


def intersection(left, right):
    a, b = merge(left), merge(right)
    i = j = 0
    total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1])-max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def require_node_graph_trace(db):
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if 'CUPTI_ACTIVITY_KIND_GRAPH_TRACE' in tables and db.execute(
            'SELECT count(*) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE').fetchone()[0]:
        raise ValueError('graph-level CUDA trace omits kernel nodes; capture --cuda-graph-trace=node before GPU activity/overlap audit')


def audit(path):
    db = sqlite3.connect(f'file:{path.resolve()}?mode=ro', uri=True)
    require_node_graph_trace(db)
    strings = dict(db.execute('SELECT id,value FROM StringIds'))
    ranges = [(start, end, text if text is not None else strings.get(text_id, ''))
        for start, end, text, text_id in db.execute('SELECT start,end,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL')]
    parents = [(a, b, n) for a, b, n in ranges if n.startswith(('optimized_', 'dataflow_')) or n == 'full_generator_late_chunk_first_pass']
    if len(parents) != 1:
        raise ValueError('exactly one explicitly profiled generator window required')
    begin, end, marker = parents[0]
    span = end-begin
    def clip(a, b):
        return max(a, begin), min(b, end)
    kernels = [(max(a, begin), min(b, end), strings.get(name, ''), stream)
        for a, b, name, stream in db.execute('SELECT start,end,demangledName,streamId FROM CUPTI_ACTIVITY_KIND_KERNEL')
        if a < end and b > begin]
    copies = [(max(a, begin), min(b, end), int(size), kind, stream)
        for a, b, size, kind, stream in db.execute('SELECT start,end,bytes,copyKind,streamId FROM CUPTI_ACTIVITY_KIND_MEMCPY')
        if a < end and b > begin]
    attention = [(a, b) for a, b, n, _ in kernels if ('flash' in n.lower() and ('fwd' in n.lower() or 'attn' in n.lower()))
        or any(tag in n for tag in ('kv_stationary_partial_kernel', 'merge_partial_kernel', 'q_stationary_page_kernel'))]
    busy = merge([(a, b) for a, b, _, _ in kernels]+[(a, b) for a, b, _, _, _ in copies])
    scoped = defaultdict(list)
    for a, b, name in ranges:
        if a < end and b > begin:
            scoped[name].append(clip(a, b))
    scopes = {name: {'range_count': len(items), 'CPU_range_wall_s': duration(items)/1e9,
        'GPU_idle_inside_range_s': (duration(items)-intersection(items, busy))/1e9,
        'fraction_of_parent': duration(items)/span} for name, items in scoped.items()}
    transfer = {}
    for kind, label in ((1, 'H2D'), (2, 'D2H'), (8, 'D2D')):
        items = [r for r in copies if r[3] == kind]
        transfer[label] = {'activity_count': len(items), 'bytes': sum(r[2] for r in items),
            'active_union_s': duration([(r[0], r[1]) for r in items])/1e9,
            'overlap_with_GPU_kernels_s': intersection([(r[0], r[1]) for r in items], [(a, b) for a, b, _, _ in kernels])/1e9}
    backend_names = [n for n in scoped if n in ('attention/grouped_fa2_complete', 'attention/route_proven_batched_fa2_complete')]
    backend_union = merge([item for n in backend_names for item in scoped[n]])
    return {'status': 'pass', 'marker': marker, 'parent_wall_s': span/1e9,
        'GPU_busy_union_s': duration(busy)/1e9, 'GPU_idle_in_parent_s': (span-duration(busy))/1e9,
        'attention_kernel_active_s': duration(attention)/1e9,
        'attention_kernel_active_fraction': duration(attention)/span,
        'complete_attention_backend_wall_s': duration(backend_union)/1e9,
        'complete_attention_backend_wall_fraction': duration(backend_union)/span,
        'CPU_scopes_not_additive': scopes, 'transfers': transfer,
        'scope': 'Nsight_measured_activity_and_CPU_launch_scopes_not_a_sum_of_service_timers',
        'GPU_idle_inside_CPU_scope_not_proof_of_CPU_compute_saturation': True,
        'sqlite_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sqlite', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    result = audit(Path(args.sqlite))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'CPU_scopes_not_additive'}, indent=2))


if __name__ == '__main__':
    main()
