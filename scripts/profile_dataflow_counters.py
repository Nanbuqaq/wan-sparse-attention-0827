#!/usr/bin/env python3
"""One complete resident backend call; Ncu excludes setup, teacher and warmup."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_dataflow_matrix import cases, selection, errors_pass
from scripts.gate_dataflow_references import PreparedFA2, teacher
from adapters.longlive_sparse.dataflow_reference import DataflowInputs, PreparedDataflow
from adapters.longlive_sparse.offline_eval import output_error_metrics


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backend', choices=('qout', 'kvout', 'fa2'), required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    case = cases()[71]
    indices, roles = selection(case)
    torch.manual_seed(case['seed'])
    full_k = case['history_frames']*1560
    k = torch.randn(12, full_k, 128, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q = torch.randn(12, case['query_tokens'], 128, device='cuda', dtype=torch.bfloat16)
    data = DataflowInputs(q, k.index_select(1, indices.cuda()), v.index_select(1, indices.cuda()), roles.cuda(), case['reuse'])
    prepared, fa2 = PreparedDataflow(data), PreparedFA2(data)
    prepared.allocate_kv_workspace()
    fn = {'qout': lambda: prepared.qout()[0], 'kvout': lambda: prepared.kvout()[0], 'fa2': fa2}[args.backend]
    reference = teacher(data)
    for _ in range(5):
        result = fn()
    torch.cuda.synchronize()
    error = output_error_metrics(reference, result)
    if not errors_pass(error):
        raise ValueError('pre-profile FP32 numerical gate failed')
    torch.cuda.cudart().cudaProfilerStart()
    with torch.cuda.nvtx.range('dataflow_counter_full_'+args.backend):
        fn()
        torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    payload = {'status': 'pass', 'backend': args.backend, 'case': case,
        'scope': 'one_complete_backend_call_after_5_warmups', 'FP32_error': error,
        'gpu': torch.cuda.get_device_name(), 'compute_capability': list(torch.cuda.get_device_capability()),
        'source_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'selected_indices_sha256': hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
        'kv_partial_workspace_bytes': prepared.workspace_bytes,
        'counter_collection_success_requires_separate_Ncu_report_audit': True,
        'profiled_time_never_enters_latency_matrix': True}
    with args.output.open('x') as handle:
        json.dump(payload, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
