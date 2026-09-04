#!/usr/bin/env python3
"""Real-CUDA gate for TransferPlan materialization and cache contracts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.history_cache import (
    CachedHistoryKV,
    HistoryKVCacheKey,
    HistoryUnionCache,
    tensor_sha256,
)
from adapters.longlive_sparse.selectors import SparseSelection
from adapters.longlive_sparse.stats import TimingBreakdown
from adapters.longlive_sparse.transfer_plan import build_transfer_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layout",
        choices=("exact_compact", "block64", "page256", "frame1560"),
        required=True,
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(20260904)
    config = SparseHistoryConfig(
        method="block64_history",
        history_density=0.25,
        block_size=64,
        pin_memory=True,
        non_blocking_h2d=True,
    )
    archive = HistoryArchive(config, spatial_height=8, spatial_width=16)
    for frame_id in (3, 7):
        key = torch.randn(
            1, 128, 2, 16, dtype=torch.bfloat16, generator=generator
        )
        value = torch.randn(
            1, 128, 2, 16, dtype=torch.bfloat16, generator=generator
        )
        archive.index_frame(0, frame_id, key, value)
    query = torch.randn(1, 192, 2, 16, generator=generator)
    route = archive.route_indexed(0, query, [3, 7], exact_k_tokens=128)
    selection = SparseSelection(
        frame_ids=route.union_frame_ids,
        token_ids=route.union_token_ids,
        scores=torch.zeros_like(route.union_frame_ids, dtype=torch.float32),
        candidate_history_tokens=route.candidate_history_tokens,
        selected_history_tokens=route.unique_history_tokens,
        candidate_units=route.candidate_history_tokens,
        selected_units=route.unique_history_tokens,
        cluster_size_min=None,
        cluster_size_max=None,
        index_bytes=0,
        timing=TimingBreakdown(),
    )
    reference = archive.materialize(
        0,
        selection,
        device="cpu",
        current_frame_id=12,
        freqs=None,
        candidate_frame_ids=[3, 7],
    )
    transfer = build_transfer_plan(
        route,
        [3, 7],
        frame_tokens=128,
        layout=args.layout,
        page_tokens=256,
        bytes_per_token=2 * 16 * torch.tensor([], dtype=torch.bfloat16).element_size(),
    )
    started = time.perf_counter()
    materialized = archive.materialize_transfer_plan(
        0,
        transfer,
        route,
        device=device,
        current_frame_id=12,
        freqs=None,
    )
    torch.cuda.synchronize(device)
    wall_s = time.perf_counter() - started
    key_error = float(
        (materialized.key.float().cpu() - reference.key.float()).abs().max()
    )
    value_error = float(
        (materialized.value.float().cpu() - reference.value.float()).abs().max()
    )
    coordinates = torch.stack(
        (route.union_frame_ids.long(), route.union_token_ids.long()), dim=-1
    )
    cache_key = HistoryKVCacheKey(
        layer_id=0,
        archive_epoch=archive.epoch,
        storage_version=archive.layer_storage_version(0),
        current_frame_id=12,
        candidate_frame_ids=(3, 7),
        selected_coordinate_sha256=tensor_sha256(coordinates),
        route_plan_sha256=route.digest(),
        rope_policy="upstream_zero",
        rope_position_sha256=tensor_sha256(materialized.positions),
        dtype=str(materialized.key.dtype),
        device=str(device),
        transfer_layout=args.layout,
        padding_strategy="rectangular_head_max",
    )
    entry = CachedHistoryKV(
        key=cache_key,
        value=materialized.value,
        key_unrotated=materialized.key_unrotated,
        key_roped=materialized.key,
        positions=materialized.positions,
        transfer_plan_sha256=transfer.digest(),
    )
    cache = HistoryUnionCache(entry.bytes + 1024)
    cache.put(entry)
    cache_hit = cache.get(cache_key) is entry
    status = "pass" if key_error == 0.0 and value_error == 0.0 and cache_hit else "fail"
    payload = {
        "status": status,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "layout": args.layout,
        "route_plan_sha256": route.digest(),
        "transfer": transfer.as_dict(),
        "materialized": {
            "wall_s": wall_s,
            "cpu_gather_s": materialized.cpu_gather_s,
            "h2d_s": materialized.h2d_s,
            "transferred_bytes": materialized.transferred_bytes,
            "payload_bytes": materialized.payload_bytes,
            "padding_bytes": materialized.padding_bytes,
            "key_max_abs": key_error,
            "value_max_abs": value_error,
        },
        "cache": cache.as_dict(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
