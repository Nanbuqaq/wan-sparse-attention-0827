#!/usr/bin/env python3
"""Same-route grouped FA2 staging pilot; no QOut/KVOut or HBM-counter claim."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.ar_routing import build_route_plan
from adapters.longlive_sparse.backends import execute_grouped_fa2
from adapters.longlive_sparse.grouped_staging import build_grouped_packing_recipe, execute_batched_grouped_fa2


def synthetic_plan(regime, queries, heads, history, exact):
    generator = torch.Generator().manual_seed(812)
    union = history if regime == 'dense' else math.floor(history*.25)
    labels = torch.arange(queries).div(64, rounding_mode='floor').view(1,1,-1).expand(1,heads,-1)
    groups = int(labels.max())+1
    selections = [[[] for _ in range(heads)]]
    for head in range(heads):
        allowed = torch.randperm(history, generator=generator)[:union].sort().values
        for group in range(groups):
            chosen = (allowed if regime in ('dense','shared') else
                      allowed[torch.randperm(union, generator=generator)[:max(1,union//8)]].sort().values)
            selections[0][head].append(chosen)
    frames = torch.arange(history).div(1560, rounding_mode='floor').view(1,1,-1).expand(1,heads,-1)
    tokens = torch.arange(history).remainder(1560).view(1,1,-1).expand(1,heads,-1)
    return build_route_plan(method='staging_pilot', routing_stage='pre-transfer', query_labels=labels,
        selections=selections, history_frame_ids=frames, history_token_ids=tokens,
        candidate_history_tokens=history, exact_k_tokens=exact, density=union/history, metadata={})


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--regime', choices=('dense','shared','fragmented','strided'), required=True)
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--repeats', type=int, default=30)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    queries, heads, history, exact, dim = (128,2,256,128,64) if args.small else (4680,12,9360,9360,128)
    plan = synthetic_plan(args.regime,queries,heads,history,exact)
    recipe = build_grouped_packing_recipe(plan,exact_tokens=exact,union_tokens=plan.union_frame_ids.shape[-1])
    result = {'source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
              'regime':args.regime,'route_sha':plan.digest(),'recipe_metadata_bytes':recipe.metadata_bytes,
              'route':plan.as_dict(),'kv_stationary_claim':False,'measured_hbm_transactions':False,
              'shape':[1,queries,history,exact,heads,dim]}
    if args.dry_run:
        result.update(status='dry_run_only',gpu_executed=False)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        torch.manual_seed(812)
        def tensor(tokens):
            value = torch.randn(1,tokens,heads,dim*(2 if args.regime=='strided' else 1),
                                device='cuda',dtype=torch.bfloat16)
            return value[...,::2] if args.regime=='strided' else value
        inputs = tuple(tensor(tokens) for tokens in (queries,exact,exact,
                           plan.union_frame_ids.shape[-1],plan.union_frame_ids.shape[-1]))
        functions = {'legacy':lambda:execute_grouped_fa2(*inputs,plan),
                     'batched_cold':lambda:execute_batched_grouped_fa2(*inputs,plan),
                     'batched_cpu_recipe_reuse':lambda:execute_batched_grouped_fa2(*inputs,plan,recipe=recipe)}
        samples = {name:[] for name in functions}
        peaks = {name:[] for name in functions}
        first_call, errors = {}, {}
        reference = None
        for name, call in functions.items():
            torch.cuda.synchronize()
            start=time.perf_counter()
            output = call()
            torch.cuda.synchronize()
            first_call[name] = time.perf_counter()-start
            if reference is None:
                reference = output.output
            errors[name]=float((reference.float()-output.output.float()).abs().max())
            assert errors[name]==0 and output.route_plan_sha256==plan.digest(), (name,errors[name])
            for _ in range(5):
                call()
        # Interleave method order to reduce thermal/order drift. No input reset
        # changes the resident Q/K/V; recipe CPU/GPU transfer is timed each call.
        for repeat in range(args.repeats):
            names=list(functions)
            names=names[repeat%len(names):]+names[:repeat%len(names)]
            for name in names:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start=time.perf_counter()
                output=functions[name]()
                torch.cuda.synchronize()
                samples[name].append(time.perf_counter()-start)
                peaks[name].append(torch.cuda.max_memory_allocated())
        result.update(status='pass',gpu_executed=True,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
            timing_scope='resident_including_cpu_recipe_metadata_H2D_staging_kernel_restore',warmup=5,
            repeats=args.repeats,first_call_including_compile_s=first_call,max_abs_vs_legacy=errors,
            variants={name:{'samples_s':values,'median_s':float(np.median(values)),
                            'p95_s':float(np.percentile(values,95)), 'peak_allocated_bytes':max(peaks[name])}
                      for name,values in samples.items()})
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='route'},indent=2))


if __name__=='__main__':
    main()
