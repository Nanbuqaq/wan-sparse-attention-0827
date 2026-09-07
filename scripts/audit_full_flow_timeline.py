#!/usr/bin/env python3
"""Attribute actual GPU activity to CPU launch scopes using CUDA correlation.

GPU kernel execution may start AFTER its Python/NVTX scope ended. Timestamp
intersection with the CPU range alone is therefore not stage attribution.
Service sums and activity unions are kept separate; memset is included.
"""
import argparse
from collections import defaultdict
import hashlib
import heapq
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_generator_timeline import duration, intersection, require_node_graph_trace


MAJOR = {"startup.load_pipeline", "input.noise", "generation.complete", "validation.latent_and_hash",
         "vae.decode_complete", "output.RGB_convert_D2H", "output.latent_save",
         "output.MP4_encode_write", "output.decode_integrity_check", "output.normalize_VAE_CPU", "output.RGB_convert_CPU"}


def attribute_runtime(scopes, runtimes):
    """Map (correlation, launch time, thread) to innermost and major CPU scope."""
    by_thread, calls = defaultdict(list), defaultdict(list)
    for start, end, name, tid in scopes:
        by_thread[tid].append((start, end, name))
    for correlation, start, tid in runtimes:
        calls[tid].append((start, correlation))
    result = {}
    for tid, entries in calls.items():
        ranges = sorted(by_thread[tid])
        active, majors, cursor = [], [], 0
        for stamp, correlation in sorted(entries):
            while cursor < len(ranges) and ranges[cursor][0] <= stamp:
                start, end, name = ranges[cursor]
                heapq.heappush(active, (-start, end, name))
                if name in MAJOR:
                    heapq.heappush(majors, (-start, end, name))
                cursor += 1
            for heap in (active, majors):
                while heap and heap[0][1] <= stamp:
                    heapq.heappop(heap)
            value = (active[0][2] if active else "unscoped", majors[0][2] if majors else "unscoped")
            if correlation in result and result[correlation] != value:
                raise ValueError("ambiguous CUDA runtime correlation; audit process/thread identity")
            result[correlation] = value
    return result


def summarize(activities):
    kernels = [(a, b) for a, b, kind, _ in activities if kind == "kernel"]
    answer = {"activity_count": len(activities), "GPU_kernel_service_sum_s": sum(b-a for a, b in kernels)/1e9,
              "GPU_kernel_active_union_s": duration(kernels)/1e9,
              "GPU_all_activity_union_s": duration([(a, b) for a, b, _, _ in activities])/1e9}
    for label in ("H2D", "D2H", "D2D", "other_copy", "memset"):
        rows = [(a, b, size) for a, b, kind, size in activities if kind == label]
        spans = [(a, b) for a, b, _ in rows]
        answer[label] = {"operations": len(rows), "bytes": sum(size for _, _, size in rows),
                         "service_sum_s": sum(b-a for a, b in spans)/1e9,
                         "active_union_s": duration(spans)/1e9,
                         "overlap_with_kernels_s": intersection(spans, kernels)/1e9}
    return answer


def audit(path):
    db = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    require_node_graph_trace(db)
    processes = db.execute("SELECT DISTINCT globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    devices = db.execute("SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    if len(processes) != 1 or len(devices) != 1:
        raise ValueError("this audit requires one traced CUDA process/device; never pool devices")
    strings = dict(db.execute("SELECT id,value FROM StringIds"))
    scopes = []
    for start, end, text, text_id, tid in db.execute("SELECT start,end,text,textId,globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL"):
        name = text if text is not None else strings.get(text_id, "")
        if name.startswith("fullflow/"):
            scopes.append((start, end, name.removeprefix("fullflow/"), tid))
    if not any(name == "generation.complete" for _, _, name, _ in scopes):
        raise ValueError("full-flow generation marker missing")
    activities = [(start, end, correlation, "kernel", 0) for start, end, correlation in
                  db.execute("SELECT start,end,correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL")]
    kinds = {1: "H2D", 2: "D2H", 8: "D2D"}
    activities += [(start, end, correlation, kinds.get(kind, "other_copy"), size) for start, end, correlation, kind, size in
                   db.execute("SELECT start,end,correlationId,copyKind,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY")]
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
        activities += [(start, end, correlation, "memset", size) for start, end, correlation, size in
                       db.execute("SELECT start,end,correlationId,bytes FROM CUPTI_ACTIVITY_KIND_MEMSET")]
    correlations = {r[2] for r in activities}
    runtimes = [(correlation, start, tid) for correlation, start, tid in
                db.execute("SELECT correlationId,start,globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME") if correlation in correlations]
    assignments = attribute_runtime(scopes, runtimes)
    leaf, major = defaultdict(list), defaultdict(list)
    for start, end, correlation, kind, size in activities:
        leaf_name, major_name = assignments.get(correlation, ("no_runtime_correlation", "no_runtime_correlation"))
        record = (start, end, kind, size)
        leaf[leaf_name].append(record)
        major[major_name].append(record)
    major_result = {}
    all_spans = [(a, b) for a, b, _, _, _ in activities]
    for name in sorted(MAJOR):
        spans = [(a, b) for a, b, n, _ in scopes if n == name]
        if not spans:
            continue
        result = summarize(major[name])
        result["CPU_range_wall_s"] = duration(spans)/1e9
        result["CPU_range_without_traced_GPU_activity_s"] = (duration(spans)-intersection(spans, all_spans))/1e9
        major_result[name] = result
    db.close()
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            sha.update(block)
    return {"status": "pass", "scope": "one_process_one_GPU_actual_activity_by_launch_correlation",
            "major_stages": major_result, "leaf_stages": {name: summarize(rows) for name, rows in sorted(leaf.items())},
            "source_sqlite_sha256": sha.hexdigest(),
            "service_sums_not_critical_path": True, "CPU_gap_not_CPU_compute_saturation": True,
            "memset_included": True, "unscoped_includes_uninstrumented_control_and_initialization": True,
            "HBM_transactions_not_measured": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.sqlite)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["major_stages"], indent=2))


if __name__ == "__main__":
    main()
