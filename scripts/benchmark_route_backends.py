#!/usr/bin/env python3
"""Warm real-shape backend benchmark replaying one immutable route plan."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.longlive_sparse.ar_routing import route_history
from adapters.longlive_sparse.backends import execute_plan
from adapters.longlive_sparse.runtime_attention import SparseHistorySelfAttention
from adapters.longlive_sparse.selectors import gather_per_head


def error_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    ref, cand = reference.float(), candidate.float()
    delta = cand - ref
    cosine = torch.nn.functional.cosine_similarity(
        ref.flatten(), cand.flatten(), dim=0
    )
    return {
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(delta.norm() / ref.norm().clamp_min(1e-12)),
        "cosine": float(cosine),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="block64_history")
    parser.add_argument("--density", type=float, default=0.25)
    parser.add_argument("--exact-k-tokens", type=int, default=9360)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    capture = torch.load(args.capture, map_location="cpu", weights_only=True)
    query = capture["query"].to(device)
    history_key = capture["key"].to(device)
    history_value = capture["value"].to(device)
    frame_ids = capture["frame_ids"].to(device)
    token_ids = capture["token_ids"].to(device)
    generator = torch.Generator(device=device).manual_seed(20260827)
    exact_key = torch.randn(
        query.shape[0],
        args.exact_k_tokens,
        query.shape[2],
        query.shape[3],
        dtype=query.dtype,
        device=device,
        generator=generator,
    )
    exact_value = torch.randn(
        exact_key.shape, dtype=query.dtype, device=device, generator=generator
    )
    route_start = time.perf_counter()
    plan = route_history(
        query,
        history_key,
        frame_ids,
        token_ids,
        method=args.method,
        density=args.density,
        exact_k_tokens=args.exact_k_tokens,
        seed=20260827,
    )
    torch.cuda.synchronize(device)
    route_ms = (time.perf_counter() - route_start) * 1000
    union_indices = SparseHistorySelfAttention._union_indices_from_coordinates(
        plan, frame_ids, token_ids
    )
    selected_key = gather_per_head(history_key, union_indices)
    selected_value = gather_per_head(history_value, union_indices)
    backends = ("grouped_fa2", "fixed64_rect", "varlen_triton")
    outputs = {}
    results = {}
    for backend in backends:
        for _ in range(args.warmup):
            execute_plan(
                backend,
                query,
                exact_key,
                exact_value,
                selected_key,
                selected_value,
                plan,
            )
        wall_values, kernel_values = [], []
        final = None
        for _ in range(args.iterations):
            start = time.perf_counter()
            final = execute_plan(
                backend,
                query,
                exact_key,
                exact_value,
                selected_key,
                selected_value,
                plan,
            )
            wall_values.append((time.perf_counter() - start) * 1000)
            kernel_values.append(final.elapsed_ms)
        outputs[backend] = final.output.detach()
        results[backend] = {
            **final.as_dict(),
            "wall_ms_median": statistics.median(wall_values),
            "wall_ms_min": min(wall_values),
            "wall_ms_max": max(wall_values),
            "backend_ms_median": statistics.median(kernel_values),
            "backend_ms_min": min(kernel_values),
            "backend_ms_max": max(kernel_values),
            "warmup": args.warmup,
            "iterations": args.iterations,
        }
    reference = outputs["grouped_fa2"]
    numerical_pass = True
    for backend in backends:
        results[backend]["error_vs_grouped"] = error_metrics(
            reference, outputs[backend]
        )
        results[backend]["same_route_plan"] = (
            results[backend]["route_plan_sha256"] == plan.digest()
        )
        metrics = results[backend]["error_vs_grouped"]
        if backend == "grouped_fa2":
            backend_pass = (
                metrics["max_abs"] <= 1e-5
                and metrics["relative_l2"] <= 1e-5
                and 1.0 - metrics["cosine"] <= 1e-6
            )
        else:
            backend_pass = (
                metrics["max_abs"] <= 0.02
                and metrics["relative_l2"] <= 0.01
                and 1.0 - metrics["cosine"] <= 0.001
            )
        results[backend]["status"] = (
            "pass"
            if results[backend]["same_route_plan"] and backend_pass
            else "fail"
        )
        numerical_pass = numerical_pass and results[backend]["status"] == "pass"
    payload = {
        "status": "pass" if numerical_pass else "fail",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "capture": str(Path(args.capture).resolve()),
        "shape": {
            "query": list(query.shape),
            "history_candidate": list(history_key.shape),
            "history_selected_union": list(selected_key.shape),
            "exact_key": list(exact_key.shape),
        },
        "method": args.method,
        "density": args.density,
        "route_ms": route_ms,
        "route": plan.as_dict(),
        "backends": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not numerical_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
