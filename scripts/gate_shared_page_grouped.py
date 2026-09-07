#!/usr/bin/env python3
"""Native FA2 paged-varlen gate with a shared exact prefix and identical KV order."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import traceback
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.ar_routing import build_route_plan
from adapters.longlive_sparse.resident_grouped import ResidentGroupedExecutor
from adapters.longlive_sparse.shared_page_grouped import SharedPageGroupedExecutor
from adapters.longlive_sparse.offline_eval import output_error_metrics, routed_history_attention


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--large', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('preserve old gate')
    torch.set_num_threads(2); torch.manual_seed(20260907)
    report = dict(status='running', scope='shared_GPU_page_storage_not_H2D_admission_or_video')
    try:
        Q,H,D,E,K = (4680,12,128,9360,9360) if args.large else (36,2,64,600,128)
        labels = (torch.arange(Q)%3).view(1,1,-1).expand(1,H,-1)
        selections = [[[torch.randperm(K)[:K//4].sort().values for _ in range(3)] for _ in range(H)]]
        plan = build_route_plan(method='page_gate',routing_stage='pre-transfer',query_labels=labels,selections=selections,
            history_frame_ids=torch.zeros(1,H,K,dtype=torch.long),history_token_ids=torch.arange(K).view(1,1,-1).expand(1,H,-1),
            candidate_history_tokens=K,exact_k_tokens=E,density=.25,metadata={})
        plan.enable_verified_digest_reuse()
        U = plan.union_frame_ids.shape[-1]
        q,ek,ev,hk,hv = [torch.randn(1,n,H,D,device='cuda',dtype=torch.bfloat16) for n in (Q,E,E,U,U)]
        teacher = routed_history_attention(q,hk,hv,plan.union_frame_ids,plan.union_token_ids,plan,
                                           exact_key=ek,exact_value=ev)
        executors = {'resident':ResidentGroupedExecutor(),'shared_pages':SharedPageGroupedExecutor()}
        values, samples = {}, {name:[] for name in executors}
        for repeat in range(35):
            for name in (('resident','shared_pages') if repeat%2 == 0 else ('shared_pages','resident')):
                torch.cuda.synchronize(); started=time.perf_counter()
                output = executors[name].execute(q,ek,ev,hk,hv,plan)
                torch.cuda.synchronize(); elapsed=time.perf_counter()-started
                values[name] = output.output
                if name == 'shared_pages':
                    accounting = output.storage_accounting
                if repeat >=5:
                    samples[name].append(elapsed)
        error = output_error_metrics(values['resident'],values['shared_pages'])
        teacher_error = output_error_metrics(teacher,values['shared_pages'])
        passed = teacher_error['max_abs'] <= .02 and teacher_error['relative_l2'] <= .01 and teacher_error['one_minus_cosine'] <= .001
        report.update(status='pass' if passed else 'negative', gpu=torch.cuda.get_device_name(), bitwise_output_equal=torch.equal(values['resident'],values['shared_pages']),
            bf16_vs_same_route_FP32=teacher_error,
            relative_to_same_route_resident=error, route_sha=plan.digest(), query_shape=list(q.shape),
            storage=accounting, samples_s=samples, median_s={k:statistics.median(v) for k,v in samples.items()})
    except BaseException:
        report.update(status='fail',traceback=traceback.format_exc()); raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as handle:
            json.dump(report,handle,indent=2);handle.write('\n')
        print(json.dumps({k:v for k,v in report.items() if k!='samples_s'}),flush=True)


if __name__ == '__main__':
    main()
