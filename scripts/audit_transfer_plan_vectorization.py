#!/usr/bin/env python3
"""Compare every plan tensor/run/SHA against the frozen scalar implementation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.ar_routing import build_route_plan
from adapters.longlive_sparse.transfer_plan import build_transfer_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    scalar_sha = '42afd3c6cb59bbb267d107a0d7fd562eca00e442'
    code = subprocess.check_output(['git', 'show', f'{scalar_sha}:adapters/longlive_sparse/transfer_plan.py'], cwd=ROOT, text=True)
    reference = types.ModuleType('adapters.longlive_sparse._frozen_plan_reference')
    reference.__package__ = 'adapters.longlive_sparse'
    sys.modules[reference.__name__] = reference
    exec(compile(code, f'{scalar_sha}/transfer_plan.py', 'exec'), reference.__dict__)
    generator = torch.Generator().manual_seed(20260912)
    rows = []
    for width in [130, 1560]:
        ids = [9, 2, 5]
        frame_ids = torch.tensor(ids).repeat_interleave(width).view(1,1,-1).expand(2,3,-1)
        token_ids = torch.arange(width).repeat(3).view(1,1,-1).expand(2,3,-1)
        selections = [[[torch.randperm(3*width, generator=generator)[:width//(h+1)].sort().values]
                       for h in range(3)] for b in range(2)]
        route = build_route_plan(method='fixture', routing_stage='pre-transfer',
            query_labels=torch.zeros(2,3,2,dtype=torch.long), selections=selections,
            history_frame_ids=frame_ids, history_token_ids=token_ids,
            candidate_history_tokens=3*width, exact_k_tokens=32, density=.25, metadata={})
        for layout in ['exact_compact', 'block64', 'page256', 'frame1560']:
            for partial in [False, True]:
                resident = ((torch.rand(route.union_frame_ids.shape, generator=generator) > .5)
                            & (route.union_frame_ids >= 0)) if partial else None
                kwargs = dict(frame_tokens=width, layout=layout, bytes_per_token=512,
                              resident_logical_mask=resident)
                times = []
                for builder in [reference.build_transfer_plan, build_transfer_plan]:
                    samples = []
                    for _ in range(3):
                        start = time.perf_counter()
                        plan = builder(route, ids, **kwargs)
                        samples.append(time.perf_counter()-start)
                    times.append(statistics.median(samples))
                    if builder == reference.build_transfer_plan:
                        expected = plan
                for name in ['physical_source_offsets','physical_counts','copy_source_offsets',
                             'copy_counts','logical_to_physical','resident_logical_mask']:
                    assert torch.equal(getattr(plan, name), getattr(expected, name)), name
                assert plan.as_dict() == expected.as_dict()
                assert plan.digest() == expected.digest()
                rows.append({'frame_tokens':width, 'layout':layout, 'partial_residency':partial,
                             'scalar_median_s':times[0], 'vectorized_median_s':times[1],
                             'speedup':times[0]/times[1], 'all_fields_and_sha_equal':True})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'status':'pass','scalar_commit':scalar_sha,
        'timing_scope':'plan construction only, CPU, background generation active','rows':rows}, indent=2)+'\n')
    print(json.dumps({'status':'pass','cases':len(rows),
                     'median_construction_speedup':statistics.median(row['speedup'] for row in rows)}))


if __name__ == '__main__':
    main()
