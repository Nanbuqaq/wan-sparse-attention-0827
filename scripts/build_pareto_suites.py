#!/usr/bin/env python3
"""Build the frozen density/prompt/RoPE/refresh/957-frame Pareto expansion."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.longlive_sparse.case_identity import build_case_identity


DENSITIES = (0.05, 0.10, 0.15, 0.20, 0.25)
REFRESH_POLICIES = ("per_chunk", "per_step")
ROPE_POLICIES = ("upstream_zero", "recency_rank", "clipped_relative_age")
BASIC_CATEGORIES = ("identity_scene", "irreversible_state")
LONG_CATEGORIES = (
    "identity_scene",
    "irreversible_state",
    "human_action",
    "fast_motion",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_token(case: dict) -> tuple:
    return (
        case["prompt_id"],
        int(case["seed"]),
        int(case["latent_frames"]),
        float(case["history_density"]),
        case["refresh_policy"],
        case["rope_policy"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-prompts", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--matrix", default="configs/gpu_case_matrix.json")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    frozen_path = Path(args.frozen_prompts)
    selection_path = Path(args.selection)
    calibration_path = Path(args.calibration)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    matrix = json.loads((ROOT / args.matrix).read_text(encoding="utf-8"))
    matrix_methods = {item["id"]: item for item in matrix["methods"]}
    if frozen.get("status") != "frozen" or frozen.get("sparse_results_used") is not False:
        raise ValueError("Pareto expansion requires the Dense-only frozen prompt manifest")
    if calibration.get("status") not in {
        "frozen_before_method_smoke",
        "frozen_before_formal_long_video",
    }:
        raise ValueError("Pareto expansion requires frozen paper parameters")
    selected_methods = list(selection.get("selected_methods", []))
    if not selected_methods:
        raise ValueError("Pareto selection is empty")
    for method in selected_methods:
        entry = matrix_methods.get(method)
        if entry is None or entry["routing_stage"] not in {
            "pre-transfer",
            "hybrid",
            "post-transfer",
        }:
            raise ValueError(f"invalid Pareto method: {method}")

    prompts = {item["category"]: item for item in frozen["prompts"]}
    missing = set((*BASIC_CATEGORIES, *LONG_CATEGORIES)) - set(prompts)
    if missing:
        raise ValueError(f"frozen prompt categories missing: {sorted(missing)}")

    sparse_cases: dict[tuple, dict] = {}

    def add_case(category: str, seed: int, latent_frames: int, density: float, refresh: str, rope: str, axis: str) -> None:
        prompt = prompts[category]
        case = {
            "prompt_id": prompt["prompt_id"],
            "category": category,
            "prompt": prompt["prompt"],
            "seed": int(seed),
            "latent_frames": int(latent_frames),
            "history_density": float(density),
            "refresh_policy": refresh,
            "rope_policy": rope,
            "backend": "grouped_fa2",
        }
        token = _case_token(case)
        if token not in sparse_cases:
            case["expansion_axes"] = []
            sparse_cases[token] = case
        if axis not in sparse_cases[token]["expansion_axes"]:
            sparse_cases[token]["expansion_axes"].append(axis)

    for category in BASIC_CATEGORIES:
        for density in DENSITIES:
            add_case(category, 20260826, 120, density, "per_chunk", "upstream_zero", "density_curve")
    for category in prompts:
        for seed in (20260826, 20260827):
            add_case(category, seed, 120, 0.25, "per_chunk", "upstream_zero", "formal_prompt_seed")
    for category in BASIC_CATEGORIES:
        for refresh in REFRESH_POLICIES:
            for rope in ROPE_POLICIES:
                add_case(category, 20260826, 120, 0.25, refresh, rope, "refresh_rope_factorial")
    for category in LONG_CATEGORIES:
        add_case(category, 20260827, 240, 0.25, "per_chunk", "upstream_zero", "long_957")

    sparse_case_list = sorted(sparse_cases.values(), key=_case_token)
    if len(sparse_case_list) != 30:
        raise RuntimeError(f"frozen expansion must contain 30 unique sparse configs, got {len(sparse_case_list)}")

    dense_cases_by = {}
    for case in sparse_case_list:
        token = (case["prompt_id"], case["seed"], case["latent_frames"])
        dense_cases_by[token] = {
            "prompt_id": case["prompt_id"],
            "category": case["category"],
            "prompt": case["prompt"],
            "seed": case["seed"],
            "latent_frames": case["latent_frames"],
        }
    dense_cases = sorted(
        dense_cases_by.values(),
        key=lambda case: (case["latent_frames"], case["prompt_id"], case["seed"]),
    )
    if len(dense_cases) != 12:
        raise RuntimeError(f"frozen expansion must contain 12 unique Dense references, got {len(dense_cases)}")

    expected = []
    dense_identity_by = {}
    for case in dense_cases:
        identity = build_case_identity(
            commit=args.commit,
            method="rag_dense",
            prompt_id=case["prompt_id"],
            prompt=case["prompt"],
            seed=case["seed"],
            latent_frames=case["latent_frames"],
            history_density=1.0,
            rope_policy="upstream_zero",
            refresh_policy="per_chunk",
            backend="packed_fa2",
        )
        record = {
            **identity,
            "method": "rag_dense",
            "routing_stage": "post-transfer",
            "backend": "packed_fa2",
            "prompt_id": case["prompt_id"],
            "seed": case["seed"],
            "latent_frames": case["latent_frames"],
            "history_density": 1.0,
            "rope_policy": "upstream_zero",
            "refresh_policy": "per_chunk",
        }
        expected.append(record)
        dense_identity_by[(case["prompt_id"], case["seed"], case["latent_frames"])] = record

    dense_reference_map = {}
    for method in selected_methods:
        entry = matrix_methods[method]
        for case in sparse_case_list:
            identity = build_case_identity(
                commit=args.commit,
                method=method,
                prompt_id=case["prompt_id"],
                prompt=case["prompt"],
                seed=case["seed"],
                latent_frames=case["latent_frames"],
                history_density=case["history_density"],
                rope_policy=case["rope_policy"],
                refresh_policy=case["refresh_policy"],
                backend=entry["backend"],
            )
            record = {
                **identity,
                "method": method,
                "routing_stage": entry["routing_stage"],
                "backend": entry["backend"],
                "prompt_id": case["prompt_id"],
                "seed": case["seed"],
                "latent_frames": case["latent_frames"],
                "history_density": case["history_density"],
                "rope_policy": case["rope_policy"],
                "refresh_policy": case["refresh_policy"],
                "expansion_axes": case["expansion_axes"],
            }
            expected.append(record)
            dense_reference_map[record["id"]] = dense_identity_by[
                (case["prompt_id"], case["seed"], case["latent_frames"])
            ]["id"]

    source_provenance = {
        "frozen_prompts": {"artifact_id": frozen_path.name, "sha256": _sha256(frozen_path)},
        "pareto_selection": {"artifact_id": selection_path.name, "sha256": _sha256(selection_path)},
        "method_params": {"artifact_id": calibration_path.name, "sha256": _sha256(calibration_path)},
    }
    suite = {
        "status": "frozen_pareto_expansion",
        "experiment_commit": args.commit,
        "source_provenance": source_provenance,
        "history_density": 0.25,
        "backend": "grouped_fa2",
        "refresh_policy": "per_chunk",
        "rope_policy": "upstream_zero",
        "record_per_call": False,
        "methods": selected_methods,
        "method_params": calibration.get("method_params", {}),
        "cases": sparse_case_list,
    }
    dense_manifest = {
        "status": "frozen_pareto_expansion",
        "experiment_commit": args.commit,
        "source_provenance": source_provenance,
        "cases": dense_cases,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rag_pareto_expansion.json").write_text(
        json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "rag_dense_pareto_expansion.json").write_text(
        json.dumps(dense_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "expected_pareto_expansion.json").write_text(
        json.dumps({"commit": args.commit, "cases": expected}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "dense_reference_map.json").write_text(
        json.dumps(dense_reference_map, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_methods": len(selected_methods),
                "sparse_configs_per_method": len(sparse_case_list),
                "dense_references": len(dense_cases),
                "expected_cases": len(expected),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
