#!/usr/bin/env python3
"""GPU routing/backend correctness gates with immutable route-plan replay.

The prepare phase serializes one exact ``HistoryRoutePlan`` and its input
tensors. Each backend is then launched in a separate process. A Triton
compiler abort therefore remains a kernel-negative result instead of erasing
the routing and other backend evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]

from adapters.longlive_sparse.ar_routing import route_history
from adapters.longlive_sparse.backends import (
    execute_fixed64_rect,
    execute_grouped_fa2,
    execute_varlen_triton,
)
from adapters.longlive_sparse.methods import METHOD_SPECS
from adapters.longlive_sparse.route_plan import HistoryRoutePlan
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.selectors import (
    SUMMARY_PRETRANSFER_METHODS,
    build_frame_index,
    route_indexed_history,
    summarize_query_for_pretransfer,
)


INITIAL_THRESHOLDS = {
    "same_fa2_max_abs": 1e-5,
    "same_fa2_relative_l2": 1e-5,
    "same_fa2_cosine_distance": 1e-6,
    "different_kernel_max_abs": 2e-2,
    "different_kernel_relative_l2": 1e-2,
    "different_kernel_cosine_distance": 1e-3,
}


def error_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.float()
    cand = candidate.float()
    delta = cand - ref
    cosine = torch.nn.functional.cosine_similarity(
        ref.flatten(), cand.flatten(), dim=0
    )
    return {
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(delta.norm() / ref.norm().clamp_min(1e-12)),
        "cosine": float(cosine),
        "cosine_distance": float(1.0 - cosine),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _make_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(20260827)
    batch, query_tokens, history_tokens, exact_tokens, heads, dim = 1, 96, 128, 64, 2, 128
    return {
        "query": torch.randn(
            batch,
            query_tokens,
            heads,
            dim,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ),
        "history_key": torch.randn(
            batch,
            history_tokens,
            heads,
            dim,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ),
        "history_value": torch.randn(
            batch,
            history_tokens,
            heads,
            dim,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ),
        "exact_key": torch.randn(
            batch,
            exact_tokens,
            heads,
            dim,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ),
        "exact_value": torch.randn(
            batch,
            exact_tokens,
            heads,
            dim,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ),
        "frame_ids": torch.arange(2, device=device)
        .repeat_interleave(64)
        .view(1, 1, -1)
        .expand(1, heads, -1),
        "token_ids": torch.arange(64, device=device)
        .repeat(2)
        .view(1, 1, -1)
        .expand(1, heads, -1),
    }


def prepare_bundle(bundle_path: Path, output_path: Path) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    tensors = _make_inputs(device)
    method_results = {}
    for method, spec in METHOD_SPECS.items():
        if method in {"native_dense", "native_block", "dense_history"}:
            continue
        try:
            if method in SUMMARY_PRETRANSFER_METHODS:
                config = SparseHistoryConfig(
                    method=method,
                    history_density=0.25,
                    block_size=64,
                    method_params={"remote_clusters": 16, "iterations": 2},
                )
                frames = []
                for frame_id in range(2):
                    start, end = frame_id * 64, (frame_id + 1) * 64
                    key = tensors["history_key"][:, start:end]
                    value = tensors["history_value"][:, start:end]
                    frames.append(
                        build_frame_index(
                            frame_id,
                            key,
                            value.detach().to("cpu"),
                            key.detach().to("cpu"),
                            config,
                            spatial_height=8,
                            spatial_width=8,
                        )
                    )
                summary = summarize_query_for_pretransfer(tensors["query"], 64)
                plan = route_indexed_history(
                    summary,
                    frames,
                    config,
                    exact_k_tokens=tensors["exact_key"].shape[1],
                )
            else:
                plan = route_history(
                    tensors["query"],
                    tensors["history_key"],
                    tensors["frame_ids"],
                    tensors["token_ids"],
                    method=method,
                    density=1.0 if method == "rag_dense" else 0.25,
                    exact_k_tokens=tensors["exact_key"].shape[1],
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
        tensors["query"],
        tensors["history_key"],
        tensors["frame_ids"],
        tensors["token_ids"],
        method="block64_history",
        density=1.0,
        exact_k_tokens=tensors["exact_key"].shape[1],
    )
    bundle = {
        "format": "longlive_gpu_gate_v1",
        "tensors": {
            name: tensor.detach().to("cpu") for name, tensor in tensors.items()
        },
        "route_plan": full_plan.state_dict(),
        "route_plan_sha256": full_plan.digest(),
    }
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, bundle_path)
    payload = {
        "status": "pass"
        if all(item["status"] == "pass" for item in method_results.values())
        else "fail",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "worktree_clean": not bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        ),
        "bundle": str(bundle_path),
        "route_plan_sha256": full_plan.digest(),
        "method_results": method_results,
    }
    _write_json(output_path, payload)
    return payload


def load_bundle(
    bundle_path: Path, device: torch.device
) -> tuple[dict[str, torch.Tensor], HistoryRoutePlan]:
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    if bundle.get("format") != "longlive_gpu_gate_v1":
        raise ValueError(f"unsupported gate bundle: {bundle.get('format')!r}")
    tensors = {
        name: tensor.to(device) for name, tensor in bundle["tensors"].items()
    }
    plan = HistoryRoutePlan.from_state_dict(bundle["route_plan"])
    if plan.digest() != bundle["route_plan_sha256"]:
        raise ValueError("route-plan SHA changed while loading the gate bundle")
    return tensors, plan


def _thresholds(path: Path | None) -> tuple[dict[str, float], str]:
    if path is None:
        return dict(INITIAL_THRESHOLDS), "initial_pre_dense_repeat"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {**INITIAL_THRESHOLDS, **payload.get("thresholds", payload)}, str(path)


def _passes(metrics: dict[str, float], thresholds: dict[str, float], prefix: str) -> bool:
    return (
        metrics["max_abs"] <= thresholds[f"{prefix}_max_abs"]
        and metrics["relative_l2"] <= thresholds[f"{prefix}_relative_l2"]
        and metrics["cosine_distance"] <= thresholds[f"{prefix}_cosine_distance"]
    )


def _packed_dense_reference(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    import flash_attn

    key = torch.cat((tensors["exact_key"], tensors["history_key"]), dim=1)
    value = torch.cat((tensors["exact_value"], tensors["history_value"]), dim=1)
    return flash_attn.flash_attn_func(
        tensors["query"], key, value, dropout_p=0.0, causal=False
    )


def run_backend(
    bundle_path: Path,
    backend: str,
    output_path: Path,
    thresholds_path: Path | None,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    tensors, plan = load_bundle(bundle_path, device)
    thresholds, threshold_source = _thresholds(thresholds_path)
    grouped_reference = execute_grouped_fa2(
        tensors["query"],
        tensors["exact_key"],
        tensors["exact_value"],
        tensors["history_key"],
        tensors["history_value"],
        plan,
    )
    if backend == "grouped_fa2":
        candidate = execute_grouped_fa2(
            tensors["query"],
            tensors["exact_key"],
            tensors["exact_value"],
            tensors["history_key"],
            tensors["history_value"],
            plan,
        )
        prefix = "same_fa2"
    elif backend == "fixed64_rect":
        candidate = execute_fixed64_rect(
            tensors["query"],
            tensors["exact_key"],
            tensors["exact_value"],
            tensors["history_key"],
            tensors["history_value"],
            plan,
        )
        prefix = "different_kernel"
    elif backend == "varlen_triton":
        candidate = execute_varlen_triton(
            tensors["query"],
            tensors["exact_key"],
            tensors["exact_value"],
            tensors["history_key"],
            tensors["history_value"],
            plan,
        )
        prefix = "different_kernel"
    else:
        raise ValueError(f"unsupported backend: {backend}")

    metrics = error_metrics(grouped_reference.output, candidate.output)
    same_route_plan = candidate.route_plan_sha256 == plan.digest()
    payload = {
        "status": "pass"
        if same_route_plan and _passes(metrics, thresholds, prefix)
        else "fail",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "backend": backend,
        "reference_backend": grouped_reference.backend,
        "threshold_source": threshold_source,
        "thresholds": thresholds,
        "same_route_plan": same_route_plan,
        "route_plan_sha256": plan.digest(),
        "candidate": candidate.as_dict(),
        "error": metrics,
    }
    if backend == "grouped_fa2":
        packed = _packed_dense_reference(tensors)
        payload["packed_fa2_full_history_error"] = error_metrics(
            packed, candidate.output
        )
        payload["dense_repeat_error"] = metrics
    _write_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("prepare", "backend", "all"), default="all"
    )
    parser.add_argument(
        "--backend",
        choices=("grouped_fa2", "fixed64_rect", "varlen_triton"),
    )
    parser.add_argument("--bundle")
    parser.add_argument("--output")
    parser.add_argument("--thresholds")
    args = parser.parse_args()

    output_root = Path(
        os.environ.get("INFER_OUTPUT_DIR", "results/metrics/gpu_gate")
    )
    bundle_path = Path(args.bundle) if args.bundle else output_root / "gate_bundle.pt"
    if args.mode == "prepare":
        output_path = (
            Path(args.output) if args.output else output_root / "routing_gate.json"
        )
        payload = prepare_bundle(bundle_path, output_path)
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit(0 if payload["status"] == "pass" else 1)
    if args.mode == "backend":
        if args.backend is None:
            parser.error("--backend is required for --mode backend")
        output_path = (
            Path(args.output)
            if args.output
            else output_root / f"backend_{args.backend}.json"
        )
        payload = run_backend(
            bundle_path,
            args.backend,
            output_path,
            Path(args.thresholds) if args.thresholds else None,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit(0 if payload["status"] == "pass" else 1)

    routing = prepare_bundle(bundle_path, output_root / "routing_gate.json")
    backend_results = {}
    for backend in ("grouped_fa2", "fixed64_rect", "varlen_triton"):
        backend_results[backend] = run_backend(
            bundle_path,
            backend,
            output_root / f"backend_{backend}.json",
            Path(args.thresholds) if args.thresholds else None,
        )
    payload = {"routing": routing, "backends": backend_results}
    _write_json(output_root / "gpu_correctness_gate.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if (
        routing["status"] != "pass"
        or backend_results["grouped_fa2"]["status"] != "pass"
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
