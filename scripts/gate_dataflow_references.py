#!/usr/bin/env python3
"""Numerical gate for actual GPU QOut/KV-stationary references and FA2."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.dataflow_reference import DataflowInputs, PreparedDataflow, StreamingDataflow
from adapters.longlive_sparse.offline_eval import output_error_metrics


def allowed_keys(tags, group, reuse):
    return torch.ones_like(tags, dtype=torch.bool) if reuse == 3 else (
        (tags == group) | ((tags+1)%3 == group) if reuse == 2 else tags == group)


@torch.inference_mode()
def teacher(c):
    h, q, d = c.query.shape
    out = torch.empty((h, q, d), dtype=torch.float32, device=c.query.device)
    groups = torch.arange(q, device=c.query.device)*3//q
    for group in range(3):
        qi = torch.nonzero(groups == group).flatten()
        ki = torch.nonzero(allowed_keys(c.key_roles, group, c.reuse)).flatten()
        if ki.numel() == 0:
            raise ValueError('every query group must retain at least one KV')
        k, v = c.key.index_select(1, ki).float(), c.value.index_select(1, ki).float()
        for start in range(0, len(qi), 64):
            ids = qi[start:start+64]
            scores = torch.bmm(c.query.index_select(1, ids).float(), k.transpose(1, 2))*d**-.5
            out.index_copy_(1, ids, torch.bmm(scores.softmax(-1), v))
    return out


class PreparedFA2:
    def __init__(self, c):
        import flash_attn
        self.fn = flash_attn.flash_attn_func
        self.output = torch.empty_like(c.query)
        self.parts = []
        self.key_indices = []
        q = c.query.shape[1]
        groups = torch.arange(q, device=c.query.device)*3//q
        for group in (range(1) if c.reuse == 3 else range(3)):
            qi = torch.arange(q, device=c.query.device) if c.reuse == 3 else torch.nonzero(groups == group).flatten()
            ki = torch.nonzero(allowed_keys(c.key_roles, group, c.reuse)).flatten()
            arrays = [x.index_select(1, ids).permute(1, 0, 2).unsqueeze(0).contiguous()
                      for x, ids in ((c.query, qi), (c.key, ki), (c.value, ki))]
            self.parts.append((qi, arrays))
            self.key_indices.append(ki)
        self.prepared_bytes = sum(a.numel()*a.element_size() for _, arrays in self.parts for a in arrays)

    def refresh_kv(self, c):
        for (_, arrays), ki in zip(self.parts, self.key_indices):
            arrays[1].copy_(c.key.index_select(1, ki).permute(1, 0, 2).unsqueeze(0))
            arrays[2].copy_(c.value.index_select(1, ki).permute(1, 0, 2).unsqueeze(0))

    def __call__(self):
        for qi, (q, k, v) in self.parts:
            value = self.fn(q, k, v, causal=False)[0].permute(1, 0, 2)
            self.output.index_copy_(1, qi, value)
        return self.output


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--large', action='store_true')
    p.add_argument('--streaming', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('real GPU required')
    torch.manual_seed(20260907)
    shapes = [(4680, 4680, 12, 128)] if args.large else [(17, 129, 2, 64), (1560, 390, 12, 128)]
    rows = []
    for q, k, h, d in shapes:
        for reuse in (1, 2, 3):
            inputs = DataflowInputs(torch.randn(h, q, d, device='cuda', dtype=torch.bfloat16),
                torch.randn(h, k, d, device='cuda', dtype=torch.bfloat16),
                torch.randn(h, k, d, device='cuda', dtype=torch.bfloat16),
                (torch.arange(k, device='cuda')//64%3).int(), reuse)
            target = teacher(inputs)
            prepared = PreparedDataflow(inputs)
            fa2 = PreparedFA2(inputs)
            for name, fn in [('qout_gpu_reference', prepared.qout), ('kv_stationary_split_reference', prepared.kvout),
                             ('grouped_fa2', lambda: (fa2(), None))]:
                started = time.perf_counter()
                value, _ = fn()
                torch.cuda.synchronize()
                error = output_error_metrics(target, value)
                passed = error['max_abs'] <= .02 and error['relative_l2'] <= .01 and error['one_minus_cosine'] <= .001
                row = {'shape': [h, q, k, d], 'reuse': reuse, 'backend': name, 'pass': passed,
                    'bf16_vs_fp32': error, 'first_call_with_compile_s': time.perf_counter()-started,
                    'kv_workspace_bytes': prepared.workspace_bytes}
                rows.append(row)
                print(json.dumps(row), flush=True)
            if args.streaming:
                # Strided CPU source selection tests that page onload is real,
                # not a reuse of the already resident reference key/value.
                indices = torch.arange(k)*2
                cpu_k = torch.zeros(h, k*2, d, dtype=torch.bfloat16)
                cpu_v = torch.zeros_like(cpu_k)
                cpu_k.index_copy_(1, indices, inputs.key.cpu())
                cpu_v.index_copy_(1, indices, inputs.value.cpu())
                stream = StreamingDataflow(inputs, cpu_k, cpu_v, indices)
                stream.kv.partial.fill_(float('nan'))
                stream.kv.lse.fill_(float('nan'))
                for backend in ('qout', 'kvout'):
                    for overlap in (False, True):
                        value = stream.run(backend, overlap=overlap)
                        torch.cuda.synchronize()
                        error = output_error_metrics(target, value)
                        passed = error['max_abs'] <= .02 and error['relative_l2'] <= .01 and error['one_minus_cosine'] <= .001
                        row = {'shape': [h, q, k, d], 'reuse': reuse, 'backend': backend,
                            'streaming': True, 'dual_stream': overlap, 'pass': passed, 'bf16_vs_fp32': error,
                            'host_pinned_bytes': stream.staging_bytes, 'page_tokens': 256,
                            'kv_h2d_bytes': 2*h*k*d*2, 'padding_bytes': 0, 'actual_overlap_proven': False}
                        rows.append(row)
                        print(json.dumps(row), flush=True)
                del stream, cpu_k, cpu_v
            del target, prepared, fa2, inputs
            torch.cuda.empty_cache()
    result = {'status': 'pass' if all(r['pass'] for r in rows) else 'fail', 'records': rows,
        'gpu': torch.cuda.get_device_name(), 'scope': 'actual_GPU_reference_correctness_not_tuned_kernel_or_video_speed'}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    if result['status'] != 'pass':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
