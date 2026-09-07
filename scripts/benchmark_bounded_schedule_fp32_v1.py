#!/usr/bin/env python3
"""Actual finite-residency H2D schedule replay, not an optimized video backend.

Each policy uses the SAME FP32 online-softmax partial backend and logical edges.
CPU archive is synthetic page-major fused K/V; upstream layout conversion is not
included. No full-history GPU shadow tensor is retained during measurements.
"""
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import statistics
import sys
import subprocess
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.bounded_schedule import actions, simulate, route_digest
from adapters.longlive_sparse.offline_eval import output_error_metrics


def groups_for(pages, reuse):
    if reuse == "high":
        return [list(range(pages)) for _ in range(3)]
    if reuse == "disjoint":
        return [[p for p in range(pages) if p % 3 == g] for g in range(3)]
    return [[p for p in range(pages) if p % 3 in (g, (g+1) % 3)] for g in range(3)]


class Replay:
    def __init__(self, query, exact, archive, groups, capacity):
        self.query, self.exact, self.archive, self.groups = query, exact, archive, groups
        self.capacity = capacity
        self.cache = torch.empty((capacity, *archive.shape[1:]), dtype=archive.dtype, device=query.device)
        self.host = [torch.empty(archive.shape[1:], dtype=archive.dtype, pin_memory=True) for _ in range(2)]
        self.compute, self.copy = torch.cuda.Stream(), torch.cuda.Stream()
        self.last_done = None

    def run(self, policy, overlap):
        if self.last_done is not None:
            self.last_done.synchronize()  # Protect pinned memory across invocations.
        default = torch.cuda.current_stream()
        compute = self.compute if overlap else default
        copy = self.copy if overlap else default
        origin = torch.cuda.Event()
        origin.record()
        compute.wait_event(origin)
        copy.wait_event(origin)
        resident = OrderedDict()
        ready = [None] * self.capacity
        consumed = [None] * self.capacity
        host_done = [None, None]
        counts = {"H2D_copy_api_calls": 0, "H2D_bytes": 0, "hits": 0, "CPU_pack_s": 0., "CPU_pinned_reuse_wait_s": 0.}
        page_bytes = self.archive[0].numel() * self.archive.element_size()
        heads, qtokens, dim = self.query.shape
        group_q = qtokens // 3
        with torch.cuda.stream(compute):
            queries = [self.query[:, g*group_q:(g+1)*group_q].float() for g in range(3)]
            states = [[torch.full((heads, group_q), -torch.inf, device=self.query.device),
                       torch.zeros((heads, group_q), device=self.query.device),
                       torch.zeros((heads, group_q, dim), device=self.query.device)] for _ in range(3)]

        def update(group, key, value):
            m, denominator, accumulator = states[group]
            scores = torch.matmul(queries[group], key.float().transpose(-1, -2)) / (dim ** .5)
            next_m = torch.maximum(m, scores.amax(-1))
            alpha = torch.exp(m - next_m)
            probability = torch.exp(scores - next_m[..., None])
            states[group] = [next_m, denominator*alpha + probability.sum(-1),
                             accumulator*alpha[..., None] + torch.matmul(probability, value.float())]

        with torch.cuda.stream(compute):
            for group in range(3):
                update(group, self.exact[0], self.exact[1])

        def fetch(block):
            if block in resident:
                slot = resident[block]
                resident.move_to_end(block)
                counts["hits"] += 1
                return slot
            if len(resident) == self.capacity:
                _, slot = resident.popitem(last=False)
            else:
                slot = len(resident)
            host_slot = counts["H2D_copy_api_calls"] % 2
            if host_done[host_slot] is not None:
                begin = time.perf_counter()
                host_done[host_slot].synchronize()
                counts["CPU_pinned_reuse_wait_s"] += time.perf_counter() - begin
            begin = time.perf_counter()
            self.host[host_slot].copy_(self.archive[block])
            counts["CPU_pack_s"] += time.perf_counter() - begin
            with torch.cuda.stream(copy):
                if consumed[slot] is not None:
                    copy.wait_event(consumed[slot])
                self.cache[slot].copy_(self.host[host_slot], non_blocking=True)
                event = torch.cuda.Event()
                event.record()
            host_done[host_slot] = ready[slot] = event
            resident[block] = slot
            counts["H2D_copy_api_calls"] += 1
            counts["H2D_bytes"] += page_bytes
            return slot

        union = sorted(set().union(*map(set, self.groups)))
        if policy == "eager_union":
            if len(union) > self.capacity:
                raise ValueError("eager union exceeds declared capacity")
            for block in union:
                fetch(block)
            # Eager history execution begins only after the complete union is
            # ready; the already-resident exact prefix may overlap its upload.
            compute.wait_event(ready[resident[union[-1]]])
        for block, neighbors in actions(self.groups, policy):
            slot = fetch(block)
            with torch.cuda.stream(compute):
                compute.wait_event(ready[slot])
                for group in neighbors:
                    update(group, self.cache[slot, 0], self.cache[slot, 1])
                event = torch.cuda.Event()
                event.record()
                consumed[slot] = event
        with torch.cuda.stream(compute):
            output = torch.cat([acc / denominator[..., None] for _, denominator, acc in states], dim=1)
            self.last_done = torch.cuda.Event()
            self.last_done.record()
        default.wait_event(self.last_done)
        expected = simulate(self.groups, policy, self.capacity)
        if counts["H2D_copy_api_calls"] != expected["misses"]:
            raise RuntimeError("actual copy API calls disagree with the independent schedule model")
        return output, counts


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=1560)
    parser.add_argument("--pages", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if args.queries % 3 or args.pages < 6 or args.pages % 3:
        raise ValueError("three nonempty groups and block-aligned fixture required")
    args.output.mkdir(parents=True, exist_ok=False)
    header = {"status": "running", "seed": 20260907, "args": vars(args) | {"output": str(args.output)},
              "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "planner_sha256": hashlib.sha256((ROOT / "adapters/longlive_sparse/bounded_schedule.py").read_bytes()).hexdigest()}
    (args.output / "manifest.json").write_text(json.dumps(header, indent=2) + "\n")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20260907)
    query = torch.randn(12, args.queries, 128, dtype=torch.bfloat16, device="cuda")
    exact = torch.randn(2, 12, 128, 128, dtype=torch.bfloat16, device="cuda")
    archive = torch.randn(args.pages, 2, 12, 64, 128, dtype=torch.bfloat16, device="cpu")
    rows = []
    for reuse in ("disjoint", "medium", "high"):
        groups = groups_for(args.pages, reuse)
        targets = []
        group_q = args.queries // 3
        for group in range(3):
            selected = archive[groups[group]].to("cuda")
            key = torch.cat((exact[0], selected[:, 0].permute(1, 0, 2, 3).flatten(1, 2)), dim=1)
            value = torch.cat((exact[1], selected[:, 1].permute(1, 0, 2, 3).flatten(1, 2)), dim=1)
            targets.append(F.scaled_dot_product_attention(query[:, group*group_q:(group+1)*group_q].float(), key.float(), value.float()))
            del selected, key, value
        target = torch.cat(targets, dim=1)
        del targets
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        for capacity in (4, args.pages):
            replay = Replay(query, exact, archive, groups, capacity)
            for policy in ("query_major", "kv_major", "windowed_union", "eager_union"):
                prediction = simulate(groups, policy, capacity)
                if prediction["status"] != "pass":
                    rows.append({"reuse": reuse, "capacity": capacity, "policy": policy, **prediction})
                    continue
                for overlap in (False, True):
                    for _ in range(args.warmup):
                        output, counts = replay.run(policy, overlap)
                        torch.cuda.synchronize()
                    error = output_error_metrics(target, output)
                    if error["max_abs"] > .02 or error["relative_l2"] > .01 or error["one_minus_cosine"] > .001:
                        raise ValueError(f"numeric gate failed: {error}")
                    samples = []
                    for _ in range(args.repeats):
                        started = time.perf_counter()
                        output, counts = replay.run(policy, overlap)
                        torch.cuda.synchronize()
                        samples.append((time.perf_counter() - started) * 1000)
                    row = {"status": "pass", "reuse": reuse, "capacity": capacity, "policy": policy,
                        "overlap_requested": overlap, "actual_overlap_proven": False, "route_sha256": route_digest(groups),
                        "median_ms": statistics.median(samples), "p95_ms": sorted(samples)[min(len(samples)-1, int(.95*len(samples)))],
                        "samples_ms": samples, "numerical_error": error, **counts,
                        "GPU_KV_backing_bytes": replay.cache.numel()*replay.cache.element_size(),
                        "host_pinned_bytes": sum(t.numel()*t.element_size() for t in replay.host)}
                    rows.append(row)
                    print(json.dumps({k:v for k,v in row.items() if k != "samples_ms"}), flush=True)
            del replay
    result = {"status": "pass", "gpu": torch.cuda.get_device_name(), "queries": args.queries, "history_pages": args.pages,
        "page_tokens": 64, "heads": 12, "head_dim": 128, "exact_resident_tokens": 128,
        "warmup": args.warmup, "repeats": args.repeats, "records": rows,
        "scope": "fixed_logical_edges_finite_capacity_H2D_scheduling_reference_not_video",
        "archive_layout": "synthetic_page_major_fused_KV; producing_this_layout_is_not_timed",
        "partial_backend": "same_FP32_online_softmax_for_every_policy_not_optimized_attention",
        "full_history_GPU_shadow_retained": False, "cache_reset_each_measured_call": True,
        "same_mathematical_attention_edges": True, "async_API_not_overlap_proof": True,
        "provenance": header | {"status": "pass"}}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
