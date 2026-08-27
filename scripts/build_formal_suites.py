#!/usr/bin/env python3
"""Freeze the two-case 477-frame base matrix after Dense-only prompt review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RAG_METHODS = [
    "rag_local",
    "random_history",
    "block64_history",
    "token_oracle",
    "kcluster32_history",
    "fixed_k128_history",
    "fixed_k256_history",
    "qlocal_kmeans8_ar",
    "radius_k256_ar",
    "qmetric_k256_r32_ar",
    "temporal_k256_t16_ar",
    "sizesplit_k128_c2_ar",
    "svg2_ar",
    "adacluster_ar",
    "svoo_ar",
    "scope_ar",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-prompts", required=True)
    parser.add_argument("--calibration")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    frozen = json.loads(Path(args.frozen_prompts).read_text(encoding="utf-8"))
    if frozen.get("sparse_results_used") is not False or frozen.get("status") != "frozen":
        raise ValueError("formal suites require a valid Dense-only frozen prompt manifest")
    by_category = {item["category"]: item for item in frozen["prompts"]}
    basic_categories = ("identity_scene", "irreversible_state")
    missing = set(basic_categories) - set(by_category)
    if missing:
        raise ValueError(f"frozen prompts missing basic categories: {sorted(missing)}")
    cases = [
        {
            "prompt_id": by_category[category]["prompt_id"],
            "category": category,
            "prompt": by_category[category]["prompt"],
            "seed": args.seed,
        }
        for category in basic_categories
    ]
    method_params = {}
    if args.calibration:
        calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        if calibration.get("status") != "frozen_before_method_smoke":
            raise ValueError("method calibration is not frozen")
        method_params = calibration.get("method_params", {})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rag_suite = {
        "status": "frozen_basic_477",
        "formal_prompts_source": str(Path(args.frozen_prompts).resolve()),
        "latent_frames": 120,
        "history_density": 0.25,
        "backend": "grouped_fa2",
        "refresh_policy": "per_chunk",
        "rope_policy": "upstream_zero",
        "record_per_call": False,
        "methods": RAG_METHODS,
        "method_params": method_params,
        "cases": cases,
    }
    native_block_manifest = {
        "status": "frozen_basic_477",
        "formal_prompts_source": str(Path(args.frozen_prompts).resolve()),
        "seeds": [args.seed],
        "candidates": [
            {key: case[key] for key in ("prompt_id", "category", "prompt")}
            for case in cases
        ],
    }
    expected = []
    for method in ("native_dense", "rag_dense", "native_block", *RAG_METHODS):
        for case in cases:
            expected.append(
                {
                    "id": f"{method}__{case['prompt_id']}__s{case['seed']}",
                    "method": method,
                    "prompt_id": case["prompt_id"],
                    "seed": case["seed"],
                }
            )
    (output_dir / "rag_basic_477.json").write_text(
        json.dumps(rag_suite, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "native_block_basic_477.json").write_text(
        json.dumps(native_block_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "expected_basic_477.json").write_text(
        json.dumps({"cases": expected}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"cases": len(cases), "expected": len(expected)}, indent=2))


if __name__ == "__main__":
    main()
