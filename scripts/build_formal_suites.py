#!/usr/bin/env python3
"""Freeze the two-case 477-frame base matrix after Dense-only prompt review."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.longlive_sparse.case_identity import (
    build_case_identity,
    resolve_experiment_commit,
)


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
    parser.add_argument("--commit")
    parser.add_argument("--matrix", default="configs/gpu_case_matrix.json")
    args = parser.parse_args()
    frozen_path = Path(args.frozen_prompts)
    calibration_path = Path(args.calibration) if args.calibration else None
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("sparse_results_used") is not False or frozen.get("status") != "frozen":
        raise ValueError("formal suites require a valid Dense-only frozen prompt manifest")
    commit = resolve_experiment_commit(args.commit, repo_root=ROOT)
    matrix_path = ROOT / args.matrix
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix_methods = {item["id"]: item for item in matrix["methods"]}
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
    calibration_source = None
    if calibration_path:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration.get("status") != "frozen_before_method_smoke":
            raise ValueError("method calibration is not frozen")
        method_params = calibration.get("method_params", {})
        calibration_source = {
            "artifact_id": calibration_path.name,
            "sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
        }
    prompt_source = {
        "artifact_id": frozen.get("artifact_id", frozen_path.name),
        "sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rag_suite = {
        "status": "frozen_basic_477",
        "experiment_commit": commit,
        "formal_prompts_source": prompt_source,
        "calibration_source": calibration_source,
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
        "experiment_commit": commit,
        "formal_prompts_source": prompt_source,
        "seeds": [args.seed],
        "candidates": [
            {key: case[key] for key in ("prompt_id", "category", "prompt")}
            for case in cases
        ],
    }
    expected = []
    for method in ("native_dense", "rag_dense", "native_block", *RAG_METHODS):
        method_entry = matrix_methods[method]
        if method == "native_dense":
            density, rope, refresh = 1.0, "not_applicable", "not_applicable"
        elif method == "rag_dense":
            density, rope, refresh = 1.0, "upstream_zero", "per_chunk"
        else:
            density, rope, refresh = 0.25, "upstream_zero", "per_chunk"
        for case in cases:
            identity = build_case_identity(
                commit=commit,
                method=method,
                prompt_id=case["prompt_id"],
                prompt=case["prompt"],
                seed=case["seed"],
                latent_frames=120,
                history_density=density,
                rope_policy=rope,
                refresh_policy=refresh,
                backend=method_entry["backend"],
            )
            expected.append(
                {
                    **identity,
                    "method": method,
                    "routing_stage": method_entry["routing_stage"],
                    "backend": method_entry["backend"],
                    "prompt_id": case["prompt_id"],
                    "seed": case["seed"],
                    "latent_frames": 120,
                    "history_density": density,
                    "rope_policy": rope,
                    "refresh_policy": refresh,
                }
            )
    (output_dir / "rag_basic_477.json").write_text(
        json.dumps(rag_suite, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "native_block_basic_477.json").write_text(
        json.dumps(native_block_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "dense_basic_477.json").write_text(
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
