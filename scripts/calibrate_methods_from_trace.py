#!/usr/bin/env python3
"""Small-grid LongLive parameter calibration on captured Q/K/V, not formal videos."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from adapters.longlive_sparse.ar_routing import route_history


GRIDS = {
    "svg2_ar": [{"q_clusters": q, "k_clusters": k, "iterations": i} for q in (100, 300) for k in (256, 512, 1000) for i in (2, 5)],
    "adacluster_ar": [{"q_clusters": q, "k_clusters": k, "threshold": t} for q in (32, 65) for k in (64, 100) for t in (4.0, 5.5, 7.0)],
    "svoo_ar": [{"q_clusters": q, "k_clusters": k, "co_cluster_iterations": i} for q in (64, 256) for k in (256, 512, 1024) for i in (1, 2)],
    "scope_ar": [{"q_clusters": q, "k_clusters": k, "top_p": p} for q in (64, 100) for k in (128, 256, 333) for p in (0.85, 0.90)],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--density", type=float, default=0.25)
    args = parser.parse_args()
    records = []
    for capture_path in args.capture:
        trace = torch.load(capture_path, map_location="cpu", weights_only=False)
        query, key = trace["query"].float(), trace["key"].float()
        for method, grid in GRIDS.items():
            for parameters in grid:
                start = time.perf_counter()
                try:
                    plan = route_history(
                        query,
                        key,
                        trace["frame_ids"],
                        trace["token_ids"],
                        method=method,
                        density=args.density,
                        exact_k_tokens=9360,
                        seed=20260827,
                        spec_override=parameters,
                    )
                    status, error = "pass", None
                except Exception as exception:
                    plan, status = None, "fail"
                    error = f"{type(exception).__name__}: {exception}"
                records.append(
                    {
                        "capture": str(Path(capture_path).resolve()),
                        "method": method,
                        "parameters": parameters,
                        "status": status,
                        "error": error,
                        "elapsed_s": time.perf_counter() - start,
                        "route": plan.as_dict() if plan else None,
                    }
                )
    payload = {
        "status": "calibration_only",
        "formal_prompts_used": False,
        "records": records,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
