#!/usr/bin/env python3
"""Actual generation/VAE GPU activity by launch correlation, not CPU-range sums."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_generator_timeline import duration, intersection, require_node_graph_trace
from scripts.audit_full_flow_timeline import attribute_runtime


def audit(path):
    db = sqlite3.connect(f'file:{path.resolve()}?mode=ro', uri=True)
    require_node_graph_trace(db)
    strings = dict(db.execute('SELECT id,value FROM StringIds'))
    scopes = [(a, b, text if text is not None else strings.get(textid, ''), tid)
        for a, b, text, textid, tid in db.execute('SELECT start,end,text,textId,globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL')]
    parents = [(a, b, name) for a, b, name, _ in scopes if name.startswith('dataflow_full_pipeline_')]
    if len(parents) != 1:
        raise ValueError('exactly one full-pipeline profiled window required')
    begin, end, marker = parents[0]
    if len(db.execute('SELECT DISTINCT globalPid,deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchall()) != 1:
        raise ValueError('one CUDA process/device required')
    runtimes = list(db.execute('SELECT correlationId,start,globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME'))
    mapping = attribute_runtime(scopes, runtimes)
    kernels = [(max(a, begin), min(b, end), corr, stream, strings.get(name, ''))
        for a, b, corr, stream, name in db.execute('SELECT start,end,correlationId,streamId,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL')
        if a < end and b > begin]
    def category(corr):
        leaf = mapping.get(corr, ('unscoped',))[0]
        return 'vae' if leaf.startswith('vae/') else ('encoder' if leaf.startswith('video/') else 'generation_or_control')
    by_stage = defaultdict(list)
    stream_ids = defaultdict(set)
    for a, b, corr, stream, _ in kernels:
        by_stage[category(corr)].append((a, b))
        stream_ids[category(corr)].add(stream)
    copies = [(max(a, begin), min(b, end), size, kind, corr)
        for a, b, size, kind, corr in db.execute('SELECT start,end,bytes,copyKind,correlationId FROM CUPTI_ACTIVITY_KIND_MEMCPY')
        if a < end and b > begin]
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    memset = [(max(a, begin), min(b, end)) for a, b in db.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMSET')
              if a < end and b > begin] if 'CUPTI_ACTIVITY_KIND_MEMSET' in tables else []
    all_kernels = [(a, b) for a, b, *_ in kernels]
    all_activity = all_kernels + [(a, b) for a, b, *_ in copies] + memset
    generation_host = [(a, b) for a, b, name, _ in scopes if name == 'streaming_pipeline/generation']
    if len(generation_host) != 1:
        raise ValueError('generation host boundary missing')
    result = dict(status='pass', marker=marker, parent_wall_s=(end-begin)/1e9,
        generation_host_wall_s=duration(generation_host)/1e9,
        GPU_active_union_s=duration(all_activity)/1e9,
        GPU_idle_in_parent_s=((end-begin)-duration(all_activity))/1e9,
        GPU_kernel_activity={stage: dict(kernel_count=len(rows), service_sum_s=sum(b-a for a,b in rows)/1e9,
            active_union_s=duration(rows)/1e9, stream_ids=sorted(stream_ids[stage])) for stage, rows in by_stage.items()},
        simultaneous_VAE_and_generation_kernels_s=intersection(by_stage['vae'], by_stage['generation_or_control'])/1e9,
        VAE_GPU_activity_during_generation_host_s=intersection(by_stage['vae'], generation_host)/1e9,
        copy_by_stage_and_direction={}, CPU_inclusive_ranges_not_additive={},
        launch_correlation_used=True, memset_included=True,
        pipeline_overlap_is_not_identical_to_simultaneous_GPU_kernels=True,
        HBM_transactions_not_measured=True, first_client_display_time_not_measured=True)
    for stage in ('vae', 'encoder', 'generation_or_control'):
        for kind, label in ((1, 'H2D'), (2, 'D2H'), (8, 'D2D')):
            rows = [(a,b,size) for a,b,size,k,c in copies if k == kind and category(c) == stage]
            if rows:
                result['copy_by_stage_and_direction'][stage+'_'+label] = dict(operations=len(rows), bytes=sum(r[2] for r in rows),
                    active_union_s=duration([(a,b) for a,b,_ in rows])/1e9,
                    actual_overlap_with_kernels_s=intersection([(a,b) for a,b,_ in rows], all_kernels)/1e9)
    for name in ('vae/stream_submit', 'video/incremental_encode', 'history/cpu_route_indexed', 'history/cpu_archive_run_pack'):
        rows = [(a,b) for a,b,n,_ in scopes if n == name]
        result['CPU_inclusive_ranges_not_additive'][name] = dict(calls=len(rows), wall_sum_s=sum(b-a for a,b in rows)/1e9)
    db.close()
    with path.open('rb') as handle:
        result['source_sqlite_sha256'] = hashlib.file_digest(handle, 'sha256').hexdigest()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sqlite', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.sqlite)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
