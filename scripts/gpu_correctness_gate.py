#!/usr/bin/env python3
"""GPU routing/backend correctness gate with immutable route-plan replay."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from adapters.longlive_sparse.ar_routing import route_history
from adapters.longlive_sparse.backends import execute_fixed64_rect, execute_grouped_fa2
from adapters.longlive_sparse.methods import METHOD_SPECS


def error_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    ref = reference.float()
    cand = candidate.float()
    delta = cand - ref
    return {
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(delta.norm() / ref.norm().clamp_min(1e-12)),
        "cosine": float(torch.nn.functional.cosine_similarity(ref.flatten(), cand.flatten(), dim=0)),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260827)
    batch, query_tokens, history_tokens, exact_tokens, heads, dim = 1, 96, 128, 64, 2, 128
    query = torch.randn(batch, query_tokens, heads, dim, dtype=torch.bfloat16, device=device, generator=generator)
    history_key = torch.randn(batch, history_tokens, heads, dim, dtype=torch.bfloat16, device=device, generator=generator)
    history_value = torch.randn(batch, history_tokens, heads, dim, dtype=torch.bfloat16, device=device, generator=generator)
    exact_key = torch.randn(batch, exact_tokens, heads, dim, dtype=torch.bfloat16, device=device, generator=generator)
    exact_value = torch.randn(batch, exact_tokens, heads, dim, dtype=torch.bfloat16, device=device, generator=generator)
    frame_ids = torch.arange(2, device=device).repeat_interleave(64).view(1, 1, -1).expand(1, heads, -1)
    token_ids = torch.arange(64, device=device).repeat(2).view(1, 1, -1).expand(1, heads, -1)
    method_results = {}
    for method, spec in METHOD_SPECS.items():
        if method in {"native_dense", "rag_dense", "dense_history"}:
            continue
        try:
            plan = route_history(
                query,
                history_key,
                frame_ids,
                token_ids,
                method=method,
                density=0.25,
                exact_k_tokens=exact_tokens,
            )
            method_results[method] = {
                "status": "pass",
                "routing_stage": spec.routing_stage,
                **plan.as_dict(),
            }
        except Exception as error:
            method_results[method] = {
                "status": "fail",
                "routing_stage": spec.routing_stage,
                "error": f"{type(error).__name__}: {error}",
            }
    full_plan = route_history(
        query,
        history_key,
        frame_ids,
        token_ids,
        method="block64_history",
        density=1.0,
        exact_k_tokens=exact_tokens,
    )
    grouped = execute_grouped_fa2(
        query, exact_key, exact_value, history_key, history_value, full_plan
    )
    backend_results = {"grouped_fa2": {"status": "pass", **grouped.as_dict()}}
    try:
        fixed = execute_fixed64_rect(
            query, exact_key, exact_value, history_key, history_value, full_plan
        )
        metrics = error_metrics(grouped.output, fixed.output)
        fixed_status = "pass" if metrics["max_abs"] <= 0.02 and metrics["relative_l2"] <= 0.01 else "fail"
        backend_results["fixed64_rect"] = {
            "status": fixed_status,
            **fixed.as_dict(),
            **metrics,
            "same_route_plan": fixed.route_plan_sha256 == grouped.route_plan_sha256,
        }
    except Exception as error:
        backend_results["fixed64_rect"] = {
            "status": "fail",
            "error": f"{type(error).__name__}: {error}",
            "route_plan_sha256": full_plan.digest(),
        }
    backend_results["varlen_triton"] = {
        "status": "fail",
        "error": "rectangular varlen Triton implementation pending",
        "route_plan_sha256": full_plan.digest(),
    }
    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "route_plan_sha256": full_plan.digest(),
        "method_results": method_results,
        "backend_results": backend_results,
    }
    output_dir = Path(os.environ.get("INFER_OUTPUT_DIR", "results/metrics/gpu_gate"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gpu_correctness_gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if backend_results["grouped_fa2"]["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

