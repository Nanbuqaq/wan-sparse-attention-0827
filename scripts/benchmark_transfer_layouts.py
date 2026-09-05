#!/usr/bin/env python3
"""Warm real-capture benchmark for layout and persistent staging choices."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.selectors import summarize_query_for_pretransfer
from adapters.longlive_sparse.staging import PinnedStagingPool
from adapters.longlive_sparse.transfer_plan import build_transfer_plan


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def _ordered_frames(frame_ids: torch.Tensor) -> list[int]:
    result = []
    for value in frame_ids[0, 0].tolist():
        value = int(value)
        if value not in result:
            result.append(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--method-params-file", default="configs/formal/method_params.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capture = torch.load(args.capture, map_location="cpu", weights_only=True)
    query, key, value = capture["query"], capture["key"], capture["value"]
    frame_ids, token_ids = capture["frame_ids"].long(), capture["token_ids"].long()
    frames = _ordered_frames(frame_ids)
    counts = [int((frame_ids[0, 0] == frame).sum()) for frame in frames]
    if not counts or len(set(counts)) != 1:
        raise ValueError("capture candidate frames must have uniform token counts")
    frame_tokens = counts[0]
    params_payload = json.loads(Path(args.method_params_file).read_text(encoding="utf-8"))
    params = params_payload["method_params"]["transfer_vaware_hybrid_history"]
    config = SparseHistoryConfig(
        method="transfer_vaware_hybrid_history",
        history_density=0.25,
        block_size=64,
        refresh_policy="per_chunk",
        rope_policy="upstream_zero",
        method_params=params,
    )
    archive = HistoryArchive(config, spatial_height=30, spatial_width=52)
    for frame in frames:
        mask = frame_ids[0, 0] == frame
        order = torch.argsort(token_ids[0, 0, mask])
        archive.index_frame(0, frame, key[:, mask][:, order], value[:, mask][:, order])
    summary = summarize_query_for_pretransfer(query, 64)
    route = archive.route_indexed(0, summary, frames, exact_k_tokens=9360)
    bytes_per_token = 2 * query.shape[-1] * query.element_size()
    device = torch.device("cuda")
    results = {}
    for layout in ("exact_compact", "block64", "page256", "frame1560"):
        transfer = build_transfer_plan(
            route,
            frames,
            frame_tokens=frame_tokens,
            layout=layout,
            page_tokens=256,
            bytes_per_token=bytes_per_token,
        )
        for staging_mode in (
            "per_call_separate",
            "persistent_separate",
            "persistent_fused",
        ):
            pool = None
            if staging_mode.startswith("persistent_"):
                pool = PinnedStagingPool(
                    slots=2,
                    budget_bytes=1024**3,
                    pin_memory=True,
                )
            for _ in range(args.warmup):
                archive.materialize_transfer_plan(
                    0,
                    transfer,
                    route,
                    device=device,
                    current_frame_id=int(capture["current_start"]) // frame_tokens,
                    freqs=None,
                    staging_pool=pool,
                    staging_mode=staging_mode,
                )
            gather_values, h2d_values, total_values = [], [], []
            transferred = padding = copies = 0
            reuse_count = 0
            for _ in range(args.iterations):
                materialized = archive.materialize_transfer_plan(
                    0,
                    transfer,
                    route,
                    device=device,
                    current_frame_id=int(capture["current_start"]) // frame_tokens,
                    freqs=None,
                    staging_pool=pool,
                    staging_mode=staging_mode,
                )
                gather_values.append(materialized.cpu_gather_s)
                h2d_values.append(materialized.h2d_s)
                total_values.append(materialized.cpu_gather_s + materialized.h2d_s)
                transferred = materialized.transferred_bytes
                padding = materialized.padding_bytes
                copies = materialized.h2d_copy_count
                reuse_count += int(materialized.staging_reused)
            key_name = f"{layout}__{staging_mode}"
            h2d_median = statistics.median(h2d_values)
            results[key_name] = {
                "layout": layout,
                "staging_mode": staging_mode,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "cpu_gather_s_median": statistics.median(gather_values),
                "cpu_gather_s_p95": _percentile(gather_values, 0.95),
                "h2d_s_median": h2d_median,
                "h2d_s_p95": _percentile(h2d_values, 0.95),
                "gather_plus_h2d_s_median": statistics.median(total_values),
                "gather_plus_h2d_s_p95": _percentile(total_values, 0.95),
                "transferred_bytes": transferred,
                "payload_bytes": transfer.payload_bytes,
                "padding_bytes": padding,
                "source_run_count": transfer.source_run_count,
                "h2d_copy_count": copies,
                "effective_h2d_bytes_per_second": transferred / h2d_median,
                "staging_reuse_fraction": reuse_count / args.iterations,
                "staging_pool": pool.as_dict() if pool is not None else None,
                "transfer_plan_sha256": transfer.digest(),
            }
    payload = {
        "status": "pass",
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "capture": str(Path(args.capture).resolve()),
        "route": route.as_dict(),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "cases": len(results), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
