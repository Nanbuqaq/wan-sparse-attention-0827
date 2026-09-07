#!/usr/bin/env python3
"""Real-GPU direct-packing gate over noncontiguous and padded grouped inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.ar_routing import build_route_plan
from adapters.longlive_sparse.backends import _sequences, execute_grouped_fa2
from adapters.longlive_sparse.resident_grouped import ResidentGroupedExecutor


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError('preserve prior gate')
    torch.set_num_threads(2)
    torch.manual_seed(20260907)
    report = dict(status='running', scope='real_GPU_packing_and_attention_gate_not_video_speedup',
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('scripts/gate_resident_grouped.py', 'adapters/longlive_sparse/resident_grouped.py',
             'adapters/longlive_sparse/grouped_pack_kernels.py')}, cases=[])
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('real CUDA required')
        for batch, heads, exact in ((1, 2, 7), (2, 3, 0), (2, 1, 11)):
            for layout in ('contiguous', 'strided_dim', 'head_major'):
                queries, history, dim = 19, 23, 64
                labels = (torch.arange(queries) % 3).view(1, 1, -1).expand(batch, heads, -1)
                selections = [[[torch.randperm(history)[:group+h+3].sort().values for group in range(3)]
                               for h in range(heads)] for _ in range(batch)]
                plan = build_route_plan(method='packing_gate', routing_stage='pre-transfer', query_labels=labels,
                    selections=selections, history_frame_ids=torch.zeros(batch, heads, history, dtype=torch.long),
                    history_token_ids=torch.arange(history).view(1, 1, -1).expand(batch, heads, -1),
                    candidate_history_tokens=history, exact_k_tokens=exact, density=.25, metadata={})
                plan.enable_verified_digest_reuse()
                union = plan.union_frame_ids.shape[-1]
                def values(tokens):
                    if layout == 'head_major':
                        return torch.randn(batch, heads, tokens, dim, dtype=torch.bfloat16, device='cuda').permute(0, 2, 1, 3)
                    x = torch.randn(batch, tokens, heads, dim*(2 if layout == 'strided_dim' else 1), dtype=torch.bfloat16, device='cuda')
                    return x[..., ::2] if layout == 'strided_dim' else x
                q, ek, ev, hk, hv = [values(n) for n in (queries, exact, exact, union, union)]
                executor = ResidentGroupedExecutor()
                recipe, transferred = executor.prepare(plan, q, ek, hk)
                packed = executor.pack(recipe, q, ek, ev, hk, hv)
                expected = _sequences(q, ek, ev, hk, hv, plan)
                for i in range(3):
                    if not torch.equal(packed[i], torch.cat([item[3+i] for item in expected]).unsqueeze(1)):
                        raise RuntimeError(f'packing mismatch: {batch, heads, exact, layout, i}')
                reference = execute_grouped_fa2(q, ek, ev, hk, hv, plan)
                output = executor.execute(q, ek, ev, hk, hv, plan)
                if not torch.equal(reference.output, output.output):
                    raise RuntimeError('same-layout FA2 outputs are not bitwise equal')
                assert output.metadata_H2D_bytes == 0 and transferred > 0
                before = executor.misses
                plan.metadata['routing_identity'] = {'changed': True}
                executor.prepare(plan, q, ek, hk)
                assert executor.misses == before+1
                report['cases'].append(dict(batch=batch, heads=heads, exact=exact, layout=layout,
                    packed_QKV_exact=True, attention_output_exact=True, metadata_hit_bytes=0,
                    invalidation_pass=True, resident_GPU_metadata_bytes=executor.GPU_metadata_bytes))
                executor.clear()
        report.update(status='pass', gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    except BaseException:
        report.update(status='fail', traceback=traceback.format_exc())
        raise
    finally:
        with args.output.open('x') as handle:
            json.dump(report, handle, indent=2)
            handle.write('\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
