#!/usr/bin/env python3
"""Full frame-pack/H2D/partial-Attention producer pilot, with finite KV storage."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.page_pipeline import BoundedPagePipeline
from adapters.longlive_sparse.offline_eval import dense_history_attention, output_error_metrics
from adapters.longlive_sparse.bounded_schedule import route_digest


class AttentionConsumer:
    def __init__(self, query, exact_key, exact_value, groups):
        from flash_attn import flash_attn_func
        self.fa2 = flash_attn_func
        self.query, self.ek, self.ev, self.groups = query, exact_key, exact_value, groups
        self.group_q = query.shape[1] // len(groups)
        self.neighbors = {page: tuple(g for g, pages in enumerate(groups) if page in pages)
                          for page in sorted(set().union(*map(set, groups)))}

    def partial(self, q, k, v):
        out, lse, debug = self.fa2(q, k, v, dropout_p=0., causal=False, return_attn_probs=True)
        if debug.numel():
            raise RuntimeError('unexpected quadratic debugging allocation')
        return out, lse

    def initialize(self):
        output, lse = self.partial(self.query, self.ek, self.ev)
        self.states = []
        for group in range(len(self.groups)):
            begin, end = group*self.group_q, (group+1)*self.group_q
            self.states.append([lse[:, :, begin:end].clone(),
                torch.ones_like(lse[:, :, begin:end]), output[:, begin:end].float()])

    def consume(self, page_id, key, value):
        for group in self.neighbors[page_id]:
            begin, end = group*self.group_q, (group+1)*self.group_q
            output, lse = self.partial(self.query[:, begin:end], key.unsqueeze(0), value.unsqueeze(0))
            maximum, denominator, accumulator = self.states[group]
            next_max = torch.maximum(maximum, lse)
            alpha, beta = torch.exp(maximum-next_max), torch.exp(lse-next_max)
            self.states[group] = [next_max, denominator*alpha+beta,
                accumulator*alpha.transpose(1, 2)[..., None]+output.float()*beta.transpose(1, 2)[..., None]]

    def finalize(self):
        return torch.cat([acc/den.transpose(1, 2)[..., None] for _, den, acc in self.states], dim=1)


class CapturedAttentionConsumer(AttentionConsumer):
    """Capture page consumption only; CPU pack/H2D remain real each invocation.

    Static Q/slot addresses are a replay prerequisite, not a general production
    interface. Compile/startup and additional graph storage are reported.
    """
    def __init__(self, query, exact_key, exact_value, groups, pipeline):
        super().__init__(query, exact_key, exact_value, groups)
        started = time.perf_counter()
        initial_memory = torch.cuda.memory_allocated(query.device)
        self.states = [[torch.zeros(1, query.shape[2], self.group_q, device=query.device),
                        torch.ones(1, query.shape[2], self.group_q, device=query.device),
                        torch.zeros(1, self.group_q, query.shape[2], query.shape[3], device=query.device)]
                       for _ in groups]
        self.graphs = {}
        signatures = {(p.count, self.neighbors[i]) for i, p in enumerate(pipeline.pages) if i in self.neighbors}
        stream = torch.cuda.Stream(device=query.device)
        # Valid data avoids generating NaNs from uninitialized page buffers
        # during warmup; no archive shadow is installed.
        pipeline.gpu.zero_()
        stream.wait_stream(torch.cuda.current_stream(query.device))
        for slot in range(pipeline.capacity):
            for count, neighbors in sorted(signatures):
                key, value = pipeline.gpu[slot, 0, :count], pipeline.gpu[slot, 1, :count]
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        self._update(neighbors, key, value)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    self._update(neighbors, key, value)
                self.graphs[(key.data_ptr(), count, neighbors)] = graph
        stream.synchronize()
        self.compile_wall_s = time.perf_counter()-started
        self.additional_graph_allocated_bytes = torch.cuda.memory_allocated(query.device)-initial_memory

    def _update(self, neighbors, key, value):
        for group in neighbors:
            begin, end = group*self.group_q, (group+1)*self.group_q
            output, lse = self.partial(self.query[:, begin:end], key.unsqueeze(0), value.unsqueeze(0))
            maximum, denominator, accumulator = self.states[group]
            next_max = torch.maximum(maximum, lse)
            alpha, beta = torch.exp(maximum-next_max), torch.exp(lse-next_max)
            next_den = denominator*alpha+beta
            next_acc = accumulator*alpha.transpose(1, 2)[..., None]+output.float()*beta.transpose(1, 2)[..., None]
            maximum.copy_(next_max)
            denominator.copy_(next_den)
            accumulator.copy_(next_acc)

    def initialize(self):
        output, lse = self.partial(self.query, self.ek, self.ev)
        for group, (maximum, denominator, accumulator) in enumerate(self.states):
            begin, end = group*self.group_q, (group+1)*self.group_q
            maximum.copy_(lse[:, :, begin:end])
            denominator.fill_(1.)
            accumulator.copy_(output[:, begin:end])

    def consume(self, page_id, key, value):
        self.graphs[(key.data_ptr(), key.shape[0], self.neighbors[page_id])].replay()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--queries', type=int, default=1560)
    parser.add_argument('--page-tokens', type=int, default=256)
    parser.add_argument('--capacity', type=int, default=4)
    parser.add_argument('--reuse', choices=('high', 'medium', 'disjoint'), default='high')
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--consumer', choices=('torch', 'cuda_graph'), default='torch')
    parser.add_argument('--profile-mode', choices=('serial', 'same_thread_async', 'producer'))
    args = parser.parse_args()
    if args.queries % 3:
        parser.error('query count must divide into three groups')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'scope': 'frame_major_CPU_pack_H2D_partial_attention_replay_not_video',
        'source_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('scripts/benchmark_page_producer.py', 'adapters/longlive_sparse/page_pipeline.py')},
        'args': vars(args) | {'output': str(args.output)}, 'warmup': 5, 'repeats': args.repeats,
        'RoPE_and_online_selection_included': False, 'full_history_GPU_shadow_during_measurement': False,
        'offline_teacher_outside_timing': True, 'GPU_rows': []}
    pipe = None
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError('real CUDA is required')
        torch.manual_seed(20260907)
        heads, dim, frame_length, frames_count, exact_length = (4, 64, 384, 2, 384) if args.smoke else (12, 128, 1560, 6, 9360)
        queries = 384 if args.smoke else args.queries
        q = torch.randn(1, queries, heads, dim, dtype=torch.bfloat16, device='cuda')
        ek, ev = [torch.randn(1, exact_length, heads, dim, dtype=torch.bfloat16, device='cuda') for _ in range(2)]
        # Original frame-major, separate K/V; fusion and last-page padding are
        # performed inside EACH measured replay, not prepared offline.
        frames = [(torch.randn(frame_length, heads, dim, dtype=torch.bfloat16),
                   torch.randn(frame_length, heads, dim, dtype=torch.bfloat16)) for _ in range(frames_count)]
        pipe = BoundedPagePipeline(frames, page_tokens=args.page_tokens, capacity=args.capacity, device=q.device)
        groups = [list(range(len(pipe.pages))) for _ in range(3)]
        if args.reuse != 'high':
            groups = [[p for p in range(len(pipe.pages)) if p % 3 in
                       ((g,) if args.reuse == 'disjoint' else (g, (g+1) % 3))] for g in range(3)]
        if any(not g for g in groups):
            raise ValueError('fixture must supply history to all query groups')
        targets = []
        group_q = queries//3
        for group, selected in enumerate(groups):
            keys, values = [], []
            for page_id in selected:
                page = pipe.pages[page_id]
                keys.append(frames[page.frame][0][page.start:page.start+page.count])
                values.append(frames[page.frame][1][page.start:page.start+page.count])
            k = torch.cat((ek, torch.cat(keys).unsqueeze(0).cuda()), dim=1)
            v = torch.cat((ev, torch.cat(values).unsqueeze(0).cuda()), dim=1)
            targets.append(dense_history_attention(q[:, group*group_q:(group+1)*group_q], k, v))
            del k, v
        reference = torch.cat(targets, dim=1)
        del targets, keys, values
        if args.consumer == 'cuda_graph':
            consumer = CapturedAttentionConsumer(q, ek, ev, groups, pipe)
            report['consumer_compile_wall_s'] = consumer.compile_wall_s
            report['consumer_graph_count'] = len(consumer.graphs)
            report['additional_graph_allocated_bytes'] = consumer.additional_graph_allocated_bytes
            # Same original eager math on the exact same page order, outside
            # timed measurements, checks that capture did not change semantics.
            eager = AttentionConsumer(q, ek, ev, groups)
            eager_output, _ = pipe.run(tuple(sorted(eager.neighbors)), eager.consume,
                mode='serial', initialize=eager.initialize, finalize=eager.finalize)
        else:
            consumer = AttentionConsumer(q, ek, ev, groups)
            eager_output = None
        order = tuple(sorted(consumer.neighbors))
        initial, samples, peak = {}, {m: [] for m in ('serial', 'same_thread_async', 'producer')}, {}
        for repeat in range(5+args.repeats):
            modes = list(samples)
            modes = modes[repeat % 3:] + modes[:repeat % 3]
            for mode in modes:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                output, metrics = pipe.run(order, consumer.consume, mode=mode,
                    initialize=consumer.initialize, finalize=consumer.finalize)
                peak[mode] = max(peak.get(mode, 0), torch.cuda.max_memory_allocated())
                if mode not in initial:
                    initial[mode] = output.clone()
                if not torch.equal(output, initial[mode]):
                    raise RuntimeError('nondeterministic output or buffer lifetime violation')
                if repeat >= 5:
                    samples[mode].append(metrics)
        error = output_error_metrics(reference, initial['producer'])
        exact_modes = all(torch.equal(initial['serial'], value) for value in initial.values())
        if eager_output is not None:
            report['bitwise_original_eager_math'] = torch.equal(eager_output, initial['producer'])
            if not report['bitwise_original_eager_math']:
                raise RuntimeError('graph capture changed original eager output')
        if not exact_modes or error['max_abs'] > .02 or error['relative_l2'] > .01 or error['one_minus_cosine'] > .001:
            raise RuntimeError(f'producer numerical gate failed: {error}, modes_exact={exact_modes}')
        if args.profile_mode:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push('dataflow_page_' + args.profile_mode)
            pipe.run(order, consumer.consume, mode=args.profile_mode,
                     initialize=consumer.initialize, finalize=consumer.finalize)
            torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
        report.update(status='pass', gpu=torch.cuda.get_device_name(), torch=torch.__version__,
            query_shape=list(q.shape), frame_lengths=[k.shape[0] for k, _ in frames],
            route_sha256=route_digest(groups), groups=groups, pages=len(pipe.pages), samples=samples,
            bf16_vs_fp32=error, all_modes_bitwise_equal=exact_modes, peak_allocated_bytes=peak,
            summary={m: {'median_s': statistics.median(r['full_wall_s'] for r in rows),
                'p95_s': float(np.percentile([r['full_wall_s'] for r in rows], 95)),
                'H2D_copy_calls': rows[0]['H2D_copy_calls'], 'H2D_payload_bytes': rows[0]['H2D_payload_bytes'],
                'H2D_padding_bytes': rows[0]['H2D_padding_bytes']} for m, rows in samples.items()})
        print(json.dumps({k: v for k, v in report.items() if k not in ('samples', 'groups')}), flush=True)
    except BaseException:
        report.update(status='fail', traceback=traceback.format_exc())
        raise
    finally:
        if pipe is not None:
            pipe.close()
        (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
