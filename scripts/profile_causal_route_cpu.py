#!/usr/bin/env python3
"""Read-only CPU route characterization on saved actual causal inputs."""
from __future__ import annotations
import argparse
import cProfile
import json
from pathlib import Path
import pstats
import statistics
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_candidate_permutation import reconstruct


def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(2)
    capture=torch.load(args.capture,map_location='cpu',weights_only=True)
    archive,summary,frames=reconstruct(capture)
    def run():return archive.route_indexed(0,summary,frames,exact_k_tokens=capture['exact_key'].shape[1])
    plan=run()
    if plan.digest()!=capture['route_sha']:raise ValueError('actual route reproduction failed')
    for _ in range(3):run()
    samples=[]
    for _ in range(20):
        start=time.perf_counter();run();samples.append(time.perf_counter()-start)
    profiler=cProfile.Profile();profiler.enable()
    for _ in range(10):run()
    profiler.disable()
    rows=[]
    for (file,line,name),(primitive,total,own,cumulative,callers) in pstats.Stats(profiler).stats.items():
        rows.append({'function':f'{file}:{line}:{name}','calls':total,'self_s':own,'cumulative_s':cumulative})
    rows.sort(key=lambda r:r['cumulative_s'],reverse=True)
    result={'status':'pass','scope':'cpu_route_only_not_end_to_end','median_s':statistics.median(samples),
        'samples_s':samples,'route_sha':plan.digest(),'query_groups_before':plan.metadata['query_groups_before_compaction'],
        'query_groups_after':plan.metadata['query_groups_after_compaction'],'profile_calls':10,'top':rows[:35]}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
