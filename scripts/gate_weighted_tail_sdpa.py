#!/usr/bin/env python3
"""GPU gate for an already prepared count-weighted prototype tail."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import traceback
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.prototype_tail import build_prototype_tail, execute_weighted_tail_sdpa
from scripts.probe_prototype_tail import weighted_attention
from adapters.longlive_sparse.offline_eval import output_error_metrics


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--large', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('preserve prior gate')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20260907)
    report = dict(status='running', operator='memory_efficient_SDPA_compact_key_bias',
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('scripts/gate_weighted_tail_sdpa.py', 'adapters/longlive_sparse/prototype_tail.py')})
    try:
        Q, F, H, D, E = (4680, 1560, 12, 128, 9360) if args.large else (24, 32, 2, 64, 32)
        K = 6*F
        q, k, v, ek, ev = [torch.randn(1, n, H, D, device='cuda', dtype=torch.bfloat16) for n in (Q, K, K, E, E)]
        indices = torch.arange(0, K, 4, device='cuda').view(1, 1, -1).expand(1, H, -1)
        tail = build_prototype_tail(k, v, indices, frame_tokens=F)
        def gather(t):
            return t.permute(0, 2, 1, 3).gather(2, indices[..., None].expand(-1, -1, -1, D)).permute(0, 2, 1, 3)
        hk, hv = gather(k), gather(v)
        fullk, fullv = torch.cat((ek, hk, tail.key), 1), torch.cat((ev, hv, tail.value), 1)
        mult = torch.cat((torch.ones(1, H, E+hk.shape[1], device='cuda'), tail.counts), -1)
        reference = torch.empty_like(q, dtype=torch.float32)
        for h in range(H):
            reference[0, :, h] = weighted_attention(q[0, :, h], fullk[0, :, h], fullv[0, :, h], mult[0, h])
        samples = []
        for repeat in range(35):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            output = execute_weighted_tail_sdpa(q, ek, ev, hk, hv, tail)
            torch.cuda.synchronize()
            if repeat >= 5:
                samples.append(time.perf_counter()-begin)
        error = output_error_metrics(reference, output)
        passed = error['max_abs'] <= .02 and error['relative_l2'] <= .01 and error['one_minus_cosine'] <= .001
        report.update(status='pass' if passed else 'negative', gpu=torch.cuda.get_device_name(), query_shape=list(q.shape),
            bf16_vs_FP32_weighted_operator=error, median_s=statistics.median(samples), samples_s=samples,
            scope='prepared_concat_bias_kernel_not_archive_onload_or_video',
            bias_dtype=str(q.dtype),
            full_QK_bias_materialized=False, explicit_math_fallback=False, prototype_bytes=tail.bytes)
        print(json.dumps({k: v for k, v in report.items() if k != 'samples_s'}), flush=True)
    except BaseException:
        report.update(status='fail', traceback=traceback.format_exc())
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as handle:
            json.dump(report, handle, indent=2)
            handle.write('\n')


if __name__ == '__main__':
    main()
