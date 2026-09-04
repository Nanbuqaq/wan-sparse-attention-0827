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
from adapters.longlive_sparse.route_plan import map_union_coordinates
from adapters.longlive_sparse.selectors import gather_per_head
from adapters.longlive_sparse.transfer_plan import build_transfer_plan
from adapters.longlive_sparse.utility import (
    query_reuse_statistics,
    route_plan_membership,
)


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


def candidate_geometry(frame_ids: torch.Tensor) -> tuple[list[int], int]:
    ordered = []
    for value in frame_ids[0, 0].detach().to("cpu").tolist():
        value = int(value)
        if not ordered or ordered[-1] != value:
            if value not in ordered:
                ordered.append(value)
    counts = [int((frame_ids[0, 0] == frame_id).sum()) for frame_id in ordered]
    if not counts or len(set(counts)) != 1:
        raise ValueError("capture candidate frames must have a uniform token count")
    return ordered, counts[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="block64_history")
    parser.add_argument("--density", type=float, default=0.25)
    parser.add_argument("--exact-k-tokens", type=int, default=9360)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--method-params-file")
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
    method_params = {}
    if args.method_params_file:
        parameter_payload = json.loads(
            Path(args.method_params_file).read_text(encoding="utf-8")
        )
        parameter_mapping = parameter_payload.get("method_params", parameter_payload)
        if args.method in parameter_mapping:
            method_params = dict(parameter_mapping[args.method])
        elif any(
            key in parameter_mapping
            for key in ("q_clusters", "k_clusters", "iterations", "top_p")
        ):
            method_params = dict(parameter_mapping)
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
        spec_override=method_params or None,
    )
    torch.cuda.synchronize(device)
    route_ms = (time.perf_counter() - route_start) * 1000
    union_indices = map_union_coordinates(plan, frame_ids, token_ids)
    selected_key = gather_per_head(history_key, union_indices)
    selected_value = gather_per_head(history_value, union_indices)
    candidate_frame_order, frame_tokens = candidate_geometry(frame_ids)
    bytes_per_token = 2 * query.shape[-1] * query.element_size()
    transfer_layouts = {}
    for layout in ("exact_compact", "block64", "page256", "frame1560"):
        transfer = build_transfer_plan(
            plan,
            candidate_frame_order,
            frame_tokens=frame_tokens,
            layout=layout,
            page_tokens=256,
            bytes_per_token=bytes_per_token,
        )
        transfer_layouts[layout] = transfer.as_dict()
    reuse = query_reuse_statistics(route_plan_membership(plan))
    backends = ("grouped_fa2", "fixed64_rect", "varlen_triton")
    outputs = {}
    results = {}
    for backend in backends:
        cold_started = time.perf_counter()
        cold = execute_plan(
            backend,
            query,
            exact_key,
            exact_value,
            selected_key,
            selected_value,
            plan,
        )
        cold_wall_ms = (time.perf_counter() - cold_started) * 1000
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
            "cold_wall_ms": cold_wall_ms,
            "cold_backend_ms": cold.elapsed_ms,
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
        "capture_metadata": {
            "layer": int(capture.get("layer", -1)),
            "current_start": int(capture.get("current_start", -1)),
        },
        "shape": {
            "query": list(query.shape),
            "history_candidate": list(history_key.shape),
            "history_selected_union": list(selected_key.shape),
            "exact_key": list(exact_key.shape),
        },
        "method": args.method,
        "method_params": method_params,
        "density": args.density,
        "route_ms": route_ms,
        "route": plan.as_dict(),
        "query_reuse": reuse,
        "transfer_layouts": transfer_layouts,
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
