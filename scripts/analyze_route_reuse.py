#!/usr/bin/env python3
"""Summarize denoising- and layer-axis reuse from captured route plans."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.longlive_sparse.reuse import route_coordinate_set, set_jaccard
from adapters.longlive_sparse.route_plan import HistoryRoutePlan


def analyze(paths: list[Path]) -> dict:
    records = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        plan = HistoryRoutePlan.from_state_dict(payload["route_plan"])
        records.append(
            {
                "path": str(path),
                "layer": int(payload["layer"]),
                "current_start": int(payload["current_start"]),
                "denoising_pass": int(payload["denoising_pass"]),
                "route_sha": plan.digest(),
                "selected_coordinate_sha256": payload.get(
                    "selected_coordinate_sha256"
                ),
                "coordinates": route_coordinate_set(plan),
                "transfer_plan_sha256": payload.get("transfer_plan_sha256"),
                "cache_hit": bool(payload.get("cache_hit", False)),
                "cache_hit_bytes": int(payload.get("cache_hit_bytes", 0)),
                "cache_miss_bytes": int(payload.get("cache_miss_bytes", 0)),
                "key_unrotated_sha256": payload.get("key_unrotated_sha256"),
                "value_sha256": payload.get("value_sha256"),
                "rope_position_sha256": payload.get("rope_position_sha256"),
                "archive_epoch": int(payload.get("archive_epoch", -1)),
                "storage_version": int(payload.get("storage_version", -1)),
            }
        )
    by_layer_chunk = defaultdict(list)
    by_chunk_pass = defaultdict(list)
    for record in records:
        by_layer_chunk[(record["layer"], record["current_start"])].append(record)
        by_chunk_pass[(record["current_start"], record["denoising_pass"])].append(record)
    denoising = []
    for (layer, current_start), rows in sorted(by_layer_chunk.items()):
        rows.sort(key=lambda item: item["denoising_pass"])
        first = rows[0]
        denoising.append(
            {
                "layer": layer,
                "current_start": current_start,
                "passes": len(rows),
                "min_jaccard_vs_first": min(
                    set_jaccard(first["coordinates"], row["coordinates"])
                    for row in rows
                ),
                "same_route_sha_all": all(
                    row["route_sha"] == first["route_sha"] for row in rows
                ),
                "cache_hits": sum(row["cache_hit"] for row in rows),
                "cache_hit_bytes": sum(row["cache_hit_bytes"] for row in rows),
                "cache_miss_bytes": sum(row["cache_miss_bytes"] for row in rows),
                "same_key_unrotated_all": (
                    all(
                        row["key_unrotated_sha256"]
                        == first["key_unrotated_sha256"]
                        for row in rows
                    )
                    if first["key_unrotated_sha256"] is not None
                    else None
                ),
                "same_value_all": (
                    all(
                        row["value_sha256"] == first["value_sha256"]
                        for row in rows
                    )
                    if first["value_sha256"] is not None
                    else None
                ),
                "same_rope_positions_all": (
                    all(
                        row["rope_position_sha256"]
                        == first["rope_position_sha256"]
                        for row in rows
                    )
                    if first["rope_position_sha256"] is not None
                    else None
                ),
                "rows": [
                    {
                        key: value
                        for key, value in row.items()
                        if key != "coordinates"
                    }
                    for row in rows
                ],
            }
        )
    layer_axis = []
    for (current_start, denoising_pass), rows in sorted(by_chunk_pass.items()):
        rows.sort(key=lambda item: item["layer"])
        adjacent = [
            {
                "left_layer": left["layer"],
                "right_layer": right["layer"],
                "jaccard": set_jaccard(left["coordinates"], right["coordinates"]),
            }
            for left, right in zip(rows, rows[1:])
        ]
        layer_axis.append(
            {
                "current_start": current_start,
                "denoising_pass": denoising_pass,
                "layers": [row["layer"] for row in rows],
                "adjacent": adjacent,
                "median_adjacent_jaccard": (
                    float(torch.tensor([item["jaccard"] for item in adjacent]).median())
                    if adjacent
                    else None
                ),
            }
        )
    return {
        "status": "pass",
        "records": len(records),
        "denoising_axis": denoising,
        "layer_axis": layer_axis,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = sorted(Path(args.capture_dir).glob("layer*_start*_pass*.pt"))
    if not paths:
        raise FileNotFoundError("no route reuse captures found")
    payload = analyze(paths)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "records": payload["records"]}, indent=2))


if __name__ == "__main__":
    main()
