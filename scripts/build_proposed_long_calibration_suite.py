#!/usr/bin/env python3
"""Build isolated 477-frame calibration suites from QKV-ranked candidates."""

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
    "coverage_cluster_history",
    "vaware_cluster_history",
    "transfer_vaware_hybrid_history",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv-calibration", required=True)
    parser.add_argument(
        "--prompts", default="configs/calibration/proposed_long_prompts.json"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--commit")
    args = parser.parse_args()

    qkv_path = Path(args.qkv_calibration)
    prompt_path = ROOT / args.prompts
    qkv = json.loads(qkv_path.read_text(encoding="utf-8"))
    prompts = json.loads(prompt_path.read_text(encoding="utf-8"))
    if qkv.get("status") != "qkv_calibrated_long_video_freeze_pending":
        raise ValueError("QKV calibration is not ready for long-video validation")
    if qkv.get("formal_prompts_used") is not False:
        raise ValueError("proposed-method calibration must not use formal prompts")
    if prompts.get("formal_prompts_used") is not False:
        raise ValueError("long calibration prompt manifest is not isolated")
    commit = resolve_experiment_commit(args.commit, repo_root=ROOT)
    cases = [
        {
            "prompt_id": candidate["prompt_id"],
            "category": candidate["category"],
            "prompt": candidate["prompt"],
            "seed": int(seed),
            "latent_frames": 120,
        }
        for candidate in prompts["candidates"]
        for seed in prompts["seeds"]
    ]
    method_params = {
        method: qkv["qkv_selected_candidates"][method]["method_params"]
        for method in METHODS
    }
    provenance = {
        "qkv_calibration": {
            "artifact_id": qkv_path.name,
            "sha256": digest(qkv_path),
        },
        "prompts": {"artifact_id": prompt_path.name, "sha256": digest(prompt_path)},
    }
    sparse_suite = {
        "status": "isolated_proposed_long_calibration",
        "formal_prompts_used": False,
        "experiment_commit": commit,
        "source_provenance": provenance,
        "latent_frames": 120,
        "history_density": 0.25,
        "backend": "grouped_fa2",
        "refresh_policy": "per_chunk",
        "rope_policy": "upstream_zero",
        "record_per_call": False,
        "methods": METHODS,
        "method_params": method_params,
        "cases": cases,
    }
    dense_manifest = {
        "status": "isolated_proposed_long_calibration",
        "formal_prompts_used": False,
        "experiment_commit": commit,
        "source_provenance": provenance,
        "cases": cases,
    }
    expected = []
    for method in ("rag_dense", *METHODS):
        for case in cases:
            density = 1.0 if method == "rag_dense" else 0.25
            backend = "packed_fa2" if method == "rag_dense" else "grouped_fa2"
            identity = build_case_identity(
                commit=commit,
                method=method,
                prompt_id=case["prompt_id"],
                prompt=case["prompt"],
                seed=case["seed"],
                latent_frames=120,
                history_density=density,
                rope_policy="upstream_zero",
                refresh_policy="per_chunk",
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
                    "latent_frames": 120,
                    "history_density": density,
                    "rope_policy": "upstream_zero",
                    "refresh_policy": "per_chunk",
                }
            )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "proposed_long_sparse.json").write_text(
        json.dumps(sparse_suite, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "proposed_long_rag_dense.json").write_text(
        json.dumps(dense_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "expected_proposed_long.json").write_text(
        json.dumps({"cases": expected}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "methods": len(METHODS),
                "calibration_cases": len(cases),
                "expected_cases": len(expected),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
