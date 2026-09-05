#!/usr/bin/env python3
"""Analyze one audited LongLive QKV capture without generating a video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.novelty import causal_prototype_novelty
from adapters.longlive_sparse.sensitivity import history_head_sensitivity
from adapters.longlive_sparse.selectors import summarize_query_for_pretransfer
from adapters.longlive_sparse.transfer_plan import build_transfer_plan
from adapters.longlive_sparse.utility import (
    VALUE_CANDIDATES,
    aggregate_value_candidate,
    compute_online_utility_proxy,
    query_reuse_statistics,
    route_plan_membership,
)


def _ordered_frame_ids(frame_ids: torch.Tensor) -> list[int]:
    result = []
    for value in frame_ids[0, 0].tolist():
        value = int(value)
        if value not in result:
            result.append(value)
    return result


def _method_params(path: str | None, method: str) -> dict:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping = payload.get("method_params", payload)
    return dict(mapping.get(method, {}))


def _summary(value: torch.Tensor) -> dict[str, float]:
    value = value.float()
    return {
        "min": float(value.min()),
        "mean": float(value.mean()),
        "max": float(value.max()),
        "std": float(value.std(unbiased=False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="transfer_vaware_hybrid_history")
    parser.add_argument("--density", type=float, default=0.25)
    parser.add_argument("--query-block-size", type=int, default=64)
    parser.add_argument("--exact-k-tokens", type=int, default=9360)
    parser.add_argument("--method-params-file")
    parser.add_argument("--sensitivity-device", choices=("none", "cpu", "cuda"), default="none")
    parser.add_argument("--sensitivity-query-chunk", type=int, default=128)
    args = parser.parse_args()
    capture_path = Path(args.capture)
    capture = torch.load(capture_path, map_location="cpu", weights_only=True)
    query = capture["query"]
    key = capture["key"]
    value = capture["value"]
    frame_ids = capture["frame_ids"].long()
    token_ids = capture["token_ids"].long()
    ordered_frames = _ordered_frame_ids(frame_ids)
    token_counts = [int((frame_ids[0, 0] == frame_id).sum()) for frame_id in ordered_frames]
    if not token_counts or len(set(token_counts)) != 1:
        raise ValueError("capture frames must have a uniform token count")
    frame_tokens = token_counts[0]
    config = SparseHistoryConfig(
        method=args.method,
        history_density=args.density,
        block_size=64,
        refresh_policy="per_chunk",
        rope_policy="upstream_zero",
        method_params=_method_params(args.method_params_file, args.method),
    )
    archive = HistoryArchive(config, spatial_height=30, spatial_width=52)
    for frame_id in ordered_frames:
        mask = frame_ids[0, 0] == frame_id
        order = torch.argsort(token_ids[0, 0, mask])
        frame_key = key[:, mask][:, order]
        frame_value = value[:, mask][:, order]
        archive.index_frame(0, frame_id, frame_key, frame_value)
    summary = summarize_query_for_pretransfer(query, args.query_block_size)
    route = archive.route_indexed(
        0, summary, ordered_frames, exact_k_tokens=args.exact_k_tokens
    )
    context = archive.online_routing_context(0, summary, ordered_frames)
    proxy = compute_online_utility_proxy(context)
    candidates = {
        name: _summary(aggregate_value_candidate(proxy.block_probabilities, name))
        for name in sorted(VALUE_CANDIDATES)
    }
    novelty = causal_prototype_novelty(
        context.key_prototypes, context.block_frame_ids
    )
    bytes_per_token = 2 * query.shape[-1] * query.element_size()
    layouts = {}
    for layout in ("exact_compact", "block64", "page256", "frame1560"):
        transfer = build_transfer_plan(
            route,
            ordered_frames,
            frame_tokens=frame_tokens,
            layout=layout,
            page_tokens=256,
            bytes_per_token=bytes_per_token,
        )
        layouts[layout] = transfer.as_dict()
    sensitivity = None
    if args.sensitivity_device != "none":
        if args.sensitivity_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA sensitivity requested but CUDA is unavailable")
        device = torch.device(args.sensitivity_device)
        sensitivity = history_head_sensitivity(
            query.to(device),
            key.to(device),
            value.to(device),
            query_chunk_size=args.sensitivity_query_chunk,
        )
    payload = {
        "status": "pass",
        "capture": str(capture_path.resolve()),
        "capture_metadata": {
            "layer": int(capture.get("layer", -1)),
            "current_start": int(capture.get("current_start", -1)),
            "query_shape": list(query.shape),
            "history_shape": list(key.shape),
            "dtype": str(query.dtype),
            "candidate_frame_ids": ordered_frames,
            "frame_tokens": frame_tokens,
        },
        "method": args.method,
        "density": args.density,
        "online_context": context.as_dict(),
        "route": route.as_dict(),
        "query_reuse": query_reuse_statistics(route_plan_membership(route)),
        "value_candidates": candidates,
        "prototype_novelty": _summary(novelty),
        "transfer_layouts": layouts,
        "head_sensitivity": sensitivity,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "output": str(output), "route_sha": route.digest()}, indent=2))


if __name__ == "__main__":
    main()
