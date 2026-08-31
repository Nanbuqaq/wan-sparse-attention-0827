#!/usr/bin/env python3
"""Build the second-seed 957-frame confirmation panel for final methods."""

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
from adapters.longlive_sparse.methods import method_spec


METHODS = [
    "native_dense",
    "native_block",
    "rag_dense",
    "block64_history",
    "transfer_vaware_hybrid_history",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-prompts", required=True)
    parser.add_argument("--method-params", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--commit")
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    prompt_path = Path(args.frozen_prompts)
    params_path = Path(args.method_params)
    frozen = json.loads(prompt_path.read_text(encoding="utf-8"))
    params = json.loads(params_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen" or frozen.get("sparse_results_used") is not False:
        raise ValueError("final long confirmation requires Dense-only frozen prompts")
    if params.get("status") != "frozen_before_formal_long_video":
        raise ValueError("proposed method parameters are not formally frozen")
    commit = resolve_experiment_commit(args.commit, repo_root=ROOT)
    cases = [
        {
            "prompt_id": item["prompt_id"],
            "category": item["category"],
            "prompt": item["prompt"],
            "seed": args.seed,
            "latent_frames": 240,
        }
        for item in frozen["prompts"]
    ]
    provenance = {
        "frozen_prompts": {"artifact_id": prompt_path.name, "sha256": sha(prompt_path)},
        "method_params": {"artifact_id": params_path.name, "sha256": sha(params_path)},
    }
    dense_manifest = {
        "status": "frozen_final_long_confirmation",
        "experiment_commit": commit,
        "source_provenance": provenance,
        "cases": cases,
    }
    sparse_suite = {
        "status": "frozen_final_long_confirmation",
        "experiment_commit": commit,
        "source_provenance": provenance,
        "history_density": 0.25,
        "backend": "grouped_fa2",
        "refresh_policy": "per_chunk",
        "rope_policy": "upstream_zero",
        "record_per_call": False,
        "methods": ["block64_history", "transfer_vaware_hybrid_history"],
        "method_params": params["method_params"],
        "cases": cases,
    }
    expected = []
    for method in METHODS:
        for case in cases:
            native = method in {"native_dense", "native_block"}
            dense = method in {"native_dense", "rag_dense"}
            density = 1.0 if dense else 0.25
            backend = "packed_fa2" if method in {"native_dense", "rag_dense"} else "grouped_fa2"
            rope = "not_applicable" if native else "upstream_zero"
            refresh = "not_applicable" if native else "per_chunk"
            identity = build_case_identity(
                commit=commit,
                method=method,
                prompt_id=case["prompt_id"],
                prompt=case["prompt"],
                seed=case["seed"],
                latent_frames=240,
                history_density=density,
                rope_policy=rope,
                refresh_policy=refresh,
                backend=backend,
            )
            expected.append(
                {
                    **identity,
                    "method": method,
                    "routing_stage": method_spec(method).routing_stage,
                    "backend": backend,
                    "prompt_id": case["prompt_id"],
                    "seed": case["seed"],
                    "latent_frames": 240,
                    "history_density": density,
                    "rope_policy": rope,
                    "refresh_policy": refresh,
                }
            )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "final_long_dense.json").write_text(
        json.dumps(dense_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "final_long_sparse.json").write_text(
        json.dumps(sparse_suite, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "expected_final_long.json").write_text(
        json.dumps({"cases": expected}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"cases": len(cases), "expected": len(expected)}, indent=2))


if __name__ == "__main__":
    main()
