#!/usr/bin/env python3
"""Offline bounded-churn audit for previous-route prefetch predictions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.longlive_sparse.prefetch import build_verified_prefetch_plan
from adapters.longlive_sparse.route_plan import HistoryRoutePlan


def _blocks(plan: HistoryRoutePlan, block_tokens: int) -> tuple[tuple[int, int, int, int], ...]:
    values = set()
    for batch in range(plan.union_frame_ids.shape[0]):
        for head in range(plan.union_frame_ids.shape[1]):
            for frame, token in zip(
                plan.union_frame_ids[batch, head], plan.union_token_ids[batch, head]
            ):
                frame_id = int(frame)
                token_id = int(token)
                if frame_id >= 0:
                    values.add((batch, head, frame_id, token_id // block_tokens))
    return tuple(sorted(values))


def analyze(
    paths: list[Path],
    *,
    block_tokens: int = 64,
    bytes_per_block: int = 32768,
    admission_multiplier: float = 1.25,
) -> dict:
    if block_tokens < 1 or bytes_per_block < 1 or admission_multiplier <= 0:
        raise ValueError("prefetch analysis parameters must be positive")
    grouped = defaultdict(list)
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        plan = HistoryRoutePlan.from_state_dict(payload["route_plan"])
        grouped[
            (int(payload["current_start"]), int(payload["denoising_pass"]))
        ].append(
            {
                "path": str(path),
                "layer": int(payload["layer"]),
                "route_sha": plan.digest(),
                "blocks": _blocks(plan, block_tokens),
            }
        )
    rows = []
    for (current_start, denoising_pass), records in sorted(grouped.items()):
        records.sort(key=lambda item: item["layer"])
        for source, target in zip(records, records[1:]):
            actual_count = len(target["blocks"])
            max_new_blocks = math.ceil(actual_count * admission_multiplier)
            plan = build_verified_prefetch_plan(
                source["blocks"],
                target["blocks"],
                resident=(),
                bytes_per_block=bytes_per_block,
                max_new_blocks=max_new_blocks,
                max_new_bytes=max_new_blocks * bytes_per_block,
            )
            rows.append(
                {
                    "current_start": current_start,
                    "denoising_pass": denoising_pass,
                    "source_layer": source["layer"],
                    "target_layer": target["layer"],
                    "source_route_sha256": source["route_sha"],
                    "actual_route_sha256": target["route_sha"],
                    "actual_blocks": actual_count,
                    "max_new_blocks": max_new_blocks,
                    "prediction_recall": plan.prediction_recall,
                    "prediction_precision": plan.prediction_precision,
                    "admitted_recall": plan.admitted_recall,
                    "admitted_precision": plan.admitted_precision,
                    "extra_bytes": plan.extra_bytes,
                    "miss_bytes": plan.miss_bytes,
                    "final_execution_exact_actual": (
                        plan.final_execution_blocks() == target["blocks"]
                    ),
                    "timeliness": None,
                    "timeliness_reason": "offline route trace has no copy-ready timeline",
                }
            )
    if not rows:
        raise ValueError("prefetch analysis needs at least one adjacent-layer pair")
    return {
        "status": "pass",
        "predictor": "previous_route",
        "records": len(rows),
        "block_tokens": block_tokens,
        "bytes_per_block": bytes_per_block,
        "admission_multiplier": admission_multiplier,
        "mean_prediction_recall": sum(row["prediction_recall"] for row in rows)
        / len(rows),
        "mean_prediction_precision": sum(
            row["prediction_precision"] for row in rows
        )
        / len(rows),
        "worst_prediction_recall": min(row["prediction_recall"] for row in rows),
        "total_extra_bytes": sum(row["extra_bytes"] for row in rows),
        "total_miss_bytes": sum(row["miss_bytes"] for row in rows),
        "all_final_execution_exact_actual": all(
            row["final_execution_exact_actual"] for row in rows
        ),
        "q_to_next_proto_status": "not_evaluated_without_aligned_next_layer_Q_and_prototype_trace",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--block-tokens", type=int, default=64)
    parser.add_argument("--bytes-per-block", type=int, default=32768)
    parser.add_argument("--admission-multiplier", type=float, default=1.25)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = sorted(Path(args.capture_dir).glob("layer*_start*_pass*.pt"))
    payload = analyze(
        paths,
        block_tokens=args.block_tokens,
        bytes_per_block=args.bytes_per_block,
        admission_multiplier=args.admission_multiplier,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "records": payload["records"]}, indent=2))


if __name__ == "__main__":
    main()
