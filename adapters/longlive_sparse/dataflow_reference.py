"""Actual GPU Q-/KV-stationary references for isolated characterization.

KV-stationary emits per-key-tile normalized partial outputs then merges them.
Its substantial workspace is explicit; this is NOT a tuned KVOut implementation
or a claim about the best possible KV-stationary kernel.
"""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _visibility(qidx, kidx, tags, Q: tl.constexpr, K: tl.constexpr, REUSE: tl.constexpr):
    groups = tl.minimum(qidx*3//Q, 2)
    valid = (qidx[:, None] < Q) & (kidx[None, :] < K)
    if REUSE == 3:
        return valid
    if REUSE == 2:
        return valid & ((groups[:, None] == tags[None, :]) | (groups[:, None] == (tags[None, :]+1)%3))
    return valid & (groups[:, None] == tags[None, :])


@triton.jit
def q_stationary_kernel(QP, KP, VP, TAG, OUT, Q: tl.constexpr, K: tl.constexpr,
                         D: tl.constexpr, REUSE: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr):
    h, qb = tl.program_id(0), tl.program_id(1)
    qi, di = qb*BQ+tl.arange(0, BQ), tl.arange(0, D)
    q = tl.load(QP+h*Q*D+qi[:, None]*D+di[None, :], qi[:, None] < Q, 0)
    m = tl.full((BQ,), float('-inf'), tl.float32)
    l = tl.zeros((BQ,), tl.float32)
    acc = tl.zeros((BQ, D), tl.float32)
    for kb in range(tl.cdiv(K, BK)):
        ki = kb*BK+tl.arange(0, BK)
        tags = tl.load(TAG+ki, ki < K, -1)
        allowed = _visibility(qi, ki, tags, Q, K, REUSE)
        if tl.sum(allowed.to(tl.int32)) > 0:
            k = tl.load(KP+h*K*D+ki[:, None]*D+di[None, :], ki[:, None] < K, 0)
            v = tl.load(VP+h*K*D+ki[:, None]*D+di[None, :], ki[:, None] < K, 0)
            scores = tl.dot(q, tl.trans(k))*(1.4426950408889634/(D**.5))
            scores = tl.where(allowed, scores, float('-inf'))
            new_m = tl.maximum(m, tl.max(scores, axis=1))
            safe_m = tl.where(new_m == float('-inf'), 0., new_m)
            alpha = tl.exp2(m-safe_m)
            p = tl.exp2(scores-safe_m[:, None])
            acc = acc*alpha[:, None]+tl.dot(p.to(v.dtype), v)
            l = l*alpha+tl.sum(p, axis=1)
            m = new_m
    output = acc/tl.maximum(l[:, None], 1.e-20)
    tl.store(OUT+h*Q*D+qi[:, None]*D+di[None, :], output, qi[:, None] < Q)


@triton.jit
def kv_stationary_partial_kernel(QP, KP, VP, TAG, PART, LSE,
    Q: tl.constexpr, K: tl.constexpr, D: tl.constexpr, REUSE: tl.constexpr,
    BQ: tl.constexpr, BK: tl.constexpr, NT: tl.constexpr, KTILE_OFFSET: tl.constexpr = 0):
    h, kb = tl.program_id(0), tl.program_id(1)
    ki, di = kb*BK+tl.arange(0, BK), tl.arange(0, D)
    # K/V load is outside the query loop: actual KV-stationary dataflow.
    k = tl.load(KP+h*K*D+ki[:, None]*D+di[None, :], ki[:, None] < K, 0)
    v = tl.load(VP+h*K*D+ki[:, None]*D+di[None, :], ki[:, None] < K, 0)
    tags = tl.load(TAG+ki, ki < K, -1)
    for qb in range(tl.cdiv(Q, BQ)):
        qi = qb*BQ+tl.arange(0, BQ)
        allowed = _visibility(qi, ki, tags, Q, K, REUSE)
        lse = tl.full((BQ,), float('-inf'), tl.float32)
        if tl.sum(allowed.to(tl.int32)) > 0:
            q = tl.load(QP+h*Q*D+qi[:, None]*D+di[None, :], qi[:, None] < Q, 0)
            scores = tl.dot(q, tl.trans(k))*(1.4426950408889634/(D**.5))
            scores = tl.where(allowed, scores, float('-inf'))
            m = tl.max(scores, axis=1)
            safe_m = tl.where(m == float('-inf'), 0., m)
            p = tl.exp2(scores-safe_m[:, None])
            l = tl.sum(p, axis=1)
            value = tl.dot(p.to(v.dtype), v)/tl.maximum(l[:, None], 1.e-20)
            lse = tl.where(l > 0., m+tl.log2(l), float('-inf'))
            # Inactive partial rows are never consumed by the masked reducer.
            tl.store(PART+((h*NT+kb+KTILE_OFFSET)*Q+qi[:, None])*D+di[None, :], value,
                     (qi[:, None] < Q) & (l[:, None] > 0.))
        tl.store(LSE+(h*NT+kb+KTILE_OFFSET)*Q+qi, lse, qi < Q)


@triton.jit
def q_stationary_page_kernel(QP, KP, VP, TAG, ACC, MAXIMUM, DENOM, OUT,
    Q: tl.constexpr, K: tl.constexpr, D: tl.constexpr, REUSE: tl.constexpr,
    BQ: tl.constexpr, BK: tl.constexpr, FIRST: tl.constexpr, LAST: tl.constexpr):
    h, qb = tl.program_id(0), tl.program_id(1)
    qi, di = qb*BQ+tl.arange(0, BQ), tl.arange(0, D)
    q = tl.load(QP+h*Q*D+qi[:, None]*D+di[None, :], qi[:, None] < Q, 0)
    if FIRST:
        m = tl.full((BQ,), float('-inf'), tl.float32)
        l = tl.zeros((BQ,), tl.float32)
        acc = tl.zeros((BQ, D), tl.float32)
    else:
        m = tl.load(MAXIMUM+h*Q+qi, qi < Q, float('-inf'))
        l = tl.load(DENOM+h*Q+qi, qi < Q, 0.)
        acc = tl.load(ACC+h*Q*D+qi[:, None]*D+di[None, :], qi[:, None] < Q, 0.)
    for kb in range(tl.cdiv(K, BK)):
        ki = kb*BK+tl.arange(0, BK)
        tags = tl.load(TAG+ki, ki < K, -1)
        allowed = _visibility(qi, ki, tags, Q, K, REUSE)
        if tl.sum(allowed.to(tl.int32)) > 0:
            k = tl.load(KP+h*K*D+ki[:, None]*D+di[None, :], ki[:, None] < K, 0)
            v = tl.load(VP+h*K*D+ki[:, None]*D+di[None, :], ki[:, None] < K, 0)
            scores = tl.dot(q, tl.trans(k))*(1.4426950408889634/(D**.5))
            scores = tl.where(allowed, scores, float('-inf'))
            new_m = tl.maximum(m, tl.max(scores, axis=1))
            safe_m = tl.where(new_m == float('-inf'), 0., new_m)
            alpha = tl.exp2(m-safe_m)
            p = tl.exp2(scores-safe_m[:, None])
            acc = acc*alpha[:, None]+tl.dot(p.to(v.dtype), v)
            l = l*alpha+tl.sum(p, axis=1)
            m = new_m
    if LAST:
        tl.store(OUT+h*Q*D+qi[:, None]*D+di[None, :], acc/tl.maximum(l[:, None], 1.e-20), qi[:, None] < Q)
    else:
        tl.store(ACC+h*Q*D+qi[:, None]*D+di[None, :], acc, qi[:, None] < Q)
        tl.store(MAXIMUM+h*Q+qi, m, qi < Q)
        tl.store(DENOM+h*Q+qi, l, qi < Q)


@triton.jit
def merge_partial_kernel(PART, LSE, OUT, Q: tl.constexpr, D: tl.constexpr,
                          NT: tl.constexpr, BQ: tl.constexpr):
    h, qb = tl.program_id(0), tl.program_id(1)
    qi, di = qb*BQ+tl.arange(0, BQ), tl.arange(0, D)
    m = tl.full((BQ,), float('-inf'), tl.float32)
    l = tl.zeros((BQ,), tl.float32)
    acc = tl.zeros((BQ, D), tl.float32)
    for kb in range(NT):
        score = tl.load(LSE+(h*NT+kb)*Q+qi, qi < Q, float('-inf'))
        new_m = tl.maximum(m, score)
        safe_m = tl.where(new_m == float('-inf'), 0., new_m)
        alpha, beta = tl.exp2(m-safe_m), tl.exp2(score-safe_m)
        part = tl.load(PART+((h*NT+kb)*Q+qi[:, None])*D+di[None, :],
                       (qi[:, None] < Q) & (score[:, None] != float('-inf')), 0.)
        acc = acc*alpha[:, None]+part*beta[:, None]
        l = l*alpha+beta
        m = new_m
    tl.store(OUT+h*Q*D+qi[:, None]*D+di[None, :], acc/tl.maximum(l[:, None], 1.e-20), qi[:, None] < Q)


@dataclass
class DataflowInputs:
    query: torch.Tensor  # H,Q,D contiguous
    key: torch.Tensor
    value: torch.Tensor
    key_roles: torch.Tensor  # K, int32; controls query-group reuse
    reuse: int

    def validate(self):
        if self.query.ndim != 3 or self.key.ndim != 3 or self.key.shape != self.value.shape:
            raise ValueError('H/Q-or-K/D tensors required')
        if self.query.shape[::2] != self.key.shape[::2] or self.query.shape[-1] not in (64, 128):
            raise ValueError('shared heads and D64/128 required')
        if self.reuse not in (1, 2, 3) or self.key_roles.shape != (self.key.shape[1],):
            raise ValueError('three-group reuse and key roles required')
        if any(not x.is_cuda or not x.is_contiguous() for x in (self.query, self.key, self.value, self.key_roles)):
            raise ValueError('resident contiguous CUDA tensors required')
        if any(x.dtype != torch.bfloat16 for x in (self.query, self.key, self.value)):
            raise ValueError('reference experiment freezes BF16 inputs')
        if self.key_roles.dtype != torch.int32 or any(x.device != self.query.device for x in (self.key, self.value, self.key_roles)):
            raise ValueError('int32 roles on the same CUDA device required')
        if min(*self.query.shape, self.key.shape[1]) < 1 or bool(((self.key_roles < 0) | (self.key_roles > 2)).any()):
            raise ValueError('nonempty shapes and role ids0..2 required')


class PreparedDataflow:
    def __init__(self, inputs):
        inputs.validate()
        self.inputs = inputs
        h, q, d = inputs.query.shape
        self.output = torch.empty_like(inputs.query)
        self.nt = triton.cdiv(inputs.key.shape[1], 64)
        self.partial = None
        self.lse = None

    def allocate_kv_workspace(self):
        if self.partial is None:
            h, q, d = self.inputs.query.shape
            self.partial = torch.empty((h, self.nt, q, d), device=self.output.device, dtype=torch.float32)
            self.lse = torch.empty((h, self.nt, q), device=self.output.device, dtype=torch.float32)

    @property
    def workspace_bytes(self):
        return 0 if self.partial is None else 4*(self.partial.numel()+self.lse.numel())

    def qout(self):
        c = self.inputs
        h, q, d = c.query.shape
        kernel = q_stationary_kernel[(h, triton.cdiv(q, 64))](c.query, c.key, c.value, c.key_roles,
            self.output, q, c.key.shape[1], d, c.reuse, 64, 64, num_warps=4, num_stages=1)
        return self.output, kernel

    def kvout(self):
        self.allocate_kv_workspace()
        c = self.inputs
        h, q, d = c.query.shape
        kernel = kv_stationary_partial_kernel[(h, self.nt)](c.query, c.key, c.value, c.key_roles,
            self.partial, self.lse, q, c.key.shape[1], d, c.reuse, 64, 64, self.nt, num_warps=4, num_stages=1)
        merge = merge_partial_kernel[(h, triton.cdiv(q, 16))](self.partial, self.lse, self.output,
            q, d, self.nt, 16, num_warps=4, num_stages=1)
        return self.output, (kernel, merge)


class StreamingDataflow:
    """Serial or dual-stream exact Page256 onload; host/device slots are fenced.

    Event spans are service/latency diagnostics, NOT proof of actual overlap.
    Nsight CUDA activity must verify kernel/copy intersections separately.
    """
    def __init__(self, inputs, cpu_key, cpu_value, selected_indices, *, page_tokens=256):
        inputs.validate()
        if page_tokens % 64 or selected_indices.numel() != inputs.key.shape[1]:
            raise ValueError('Page256/Block64-aligned page geometry required')
        self.inputs, self.cpu_key, self.cpu_value = inputs, cpu_key, cpu_value
        self.selected = selected_indices.cpu().long()
        self.page_tokens = page_tokens
        h, q, d = inputs.query.shape
        elements = h*page_tokens*d
        self.host = [(torch.empty(elements, dtype=inputs.key.dtype, pin_memory=True),
                      torch.empty(elements, dtype=inputs.key.dtype, pin_memory=True)) for _ in range(2)]
        self.device = [(torch.empty(elements, dtype=inputs.key.dtype, device=inputs.query.device),
                        torch.empty(elements, dtype=inputs.key.dtype, device=inputs.query.device)) for _ in range(2)]
        self.compute_stream, self.copy_stream = torch.cuda.Stream(), torch.cuda.Stream()
        self.kv = PreparedDataflow(inputs)
        self.kv.allocate_kv_workspace()
        self.acc = torch.empty((h, q, d), dtype=torch.float32, device=inputs.query.device)
        self.maximum = torch.empty((h, q), dtype=torch.float32, device=inputs.query.device)
        self.denom = torch.empty_like(self.maximum)
        self.output = self.kv.output
        self.last_events = None
        self._previous_host_copies = [None, None]

    @property
    def staging_bytes(self):
        return sum(t.numel()*t.element_size() for pair in self.host for t in pair)

    def run(self, backend, *, overlap=False, record_events=False):
        if backend not in ('qout', 'kvout'):
            raise ValueError('unknown streaming dataflow')
        c = self.inputs
        h, q, d = c.query.shape
        total = c.key.shape[1]
        origin = torch.cuda.Event(enable_timing=True)
        origin.record()
        default = torch.cuda.current_stream()
        compute = self.compute_stream if overlap else default
        copy = self.copy_stream if overlap else default
        compute.wait_event(origin)
        copy.wait_event(origin)
        copied, consumed = list(self._previous_host_copies), [None, None]
        intervals = []
        for page, start in enumerate(range(0, total, self.page_tokens)):
            end = min(start+self.page_tokens, total)
            width, slot = end-start, page%2
            # A pinned host buffer may not be overwritten before its DMA ends.
            if copied[slot] is not None:
                copied[slot].synchronize()
            host_k, host_v = [x[:h*width*d].view(h, width, d) for x in self.host[slot]]
            gpu_k, gpu_v = [x[:h*width*d].view(h, width, d) for x in self.device[slot]]
            torch.index_select(self.cpu_key, 1, self.selected[start:end], out=host_k)
            torch.index_select(self.cpu_value, 1, self.selected[start:end], out=host_v)
            c0 = torch.cuda.Event(enable_timing=True) if record_events else None
            a0 = torch.cuda.Event(enable_timing=True) if record_events else None
            c1, a1 = torch.cuda.Event(enable_timing=record_events), torch.cuda.Event(enable_timing=record_events)
            with torch.cuda.stream(copy):
                if consumed[slot] is not None:
                    copy.wait_event(consumed[slot])
                if c0 is not None:
                    c0.record()
                gpu_k.copy_(host_k, non_blocking=True)
                gpu_v.copy_(host_v, non_blocking=True)
                c1.record()
            with torch.cuda.stream(compute):
                compute.wait_event(c1)
                if a0 is not None:
                    a0.record()
                if backend == 'qout':
                    q_stationary_page_kernel[(h, triton.cdiv(q, 64))](c.query, gpu_k, gpu_v,
                        c.key_roles[start:end], self.acc, self.maximum, self.denom, self.output,
                        q, width, d, c.reuse, 64, 64, start == 0, end == total, num_warps=4, num_stages=1)
                else:
                    kv_stationary_partial_kernel[(h, triton.cdiv(width, 64))](c.query, gpu_k, gpu_v,
                        c.key_roles[start:end], self.kv.partial, self.kv.lse,
                        q, width, d, c.reuse, 64, 64, self.kv.nt, start//64, num_warps=4, num_stages=1)
                a1.record()
            copied[slot], consumed[slot] = c1, a1
            if record_events:
                intervals.append((c0, c1, a0, a1))
        with torch.cuda.stream(compute):
            if backend == 'kvout':
                merge_partial_kernel[(h, triton.cdiv(q, 16))](self.kv.partial, self.kv.lse, self.output,
                    q, d, self.kv.nt, 16, num_warps=4, num_stages=1)
            done = torch.cuda.Event()
            done.record()
        default.wait_event(done)
        self._previous_host_copies = copied
        self.last_events = (origin, intervals) if record_events else None
        return self.output
