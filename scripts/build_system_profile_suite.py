#!/usr/bin/env python3
"""Build the gated 39/120/240-latent system profile matrix."""

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
from adapters.longlive_sparse.formal_gate import load_frozen_system_holdouts
from adapters.longlive_sparse.methods import method_spec
from adapters.longlive_sparse.system_config import LongLiveSystemConfig


SYSTEM_CONFIGS = {
    "legacy": LongLiveSystemConfig(profile_mode="trace"),
    "exact_per_chunk_roped": LongLiveSystemConfig(
        profile_mode="trace",
        transfer_layout="exact_compact",
        gpu_union_cache="per_chunk",
        gpu_union_cache_budget_mib=768,
        cache_payload="roped_kv",
        staging_mode="persistent_separate",
        pinned_buffer_slots=2,
        host_pinned_budget_mib=1024,
    ),
    "block64_per_chunk_roped": LongLiveSystemConfig(
        profile_mode="trace",
        transfer_layout="block64",
        gpu_union_cache="per_chunk",
        gpu_union_cache_budget_mib=768,
        cache_payload="roped_kv",
        staging_mode="persistent_separate",
        pinned_buffer_slots=2,
        host_pinned_budget_mib=1024,
    ),
    "exact_cross_chunk_raw": LongLiveSystemConfig(
        profile_mode="trace",
        transfer_layout="exact_compact",
        gpu_union_cache="cross_chunk",
        gpu_union_cache_budget_mib=768,
        cache_payload="raw_kv",
        staging_mode="per_call_separate",
        pinned_buffer_slots=2,
        host_pinned_budget_mib=1024,
    ),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    *,
    holdout_path: Path,
    prompt_path: Path,
    method_params_path: Path,
    commit: str,
) -> tuple[dict, dict]:
    holdouts = load_frozen_system_holdouts(holdout_path)
    prompts = json.loads(prompt_path.read_text(encoding="utf-8"))
    if prompts.get("status") != "isolated_non_formal_calibration":
        raise ValueError("system profile prompts are not isolated calibration data")
    if prompts.get("formal_prompts_used") is not False:
        raise ValueError("system profile cannot use formal prompts")
    if prompts.get("seeds") != [20260904]:
        raise ValueError("system profile seed must be frozen to 20260904")
    motion = next(
        item for item in prompts["candidates"] if item["prompt_id"] == "calibration_motion"
    )
    params = json.loads(method_params_path.read_text(encoding="utf-8"))[
        "method_params"
    ]["transfer_vaware_hybrid_history"]
    cases = []
    expected = []
    for system_id, system_config in SYSTEM_CONFIGS.items():
        for latent_frames in (39, 120, 240):
            case = {
                "profile_config_id": system_id,
                "prompt_id": motion["prompt_id"],
                "category": motion["category"],
                "prompt": motion["prompt"],
                "seed": 20260904,
                "latent_frames": latent_frames,
                "record_per_call": latent_frames == 39,
                "longlive_system": system_config.as_dict(),
            }
            cases.append(case)
            identity = build_case_identity(
                commit=commit,
                method="transfer_vaware_hybrid_history",
                prompt_id=case["prompt_id"],
                prompt=case["prompt"],
                seed=case["seed"],
                latent_frames=latent_frames,
                history_density=0.25,
                rope_policy="upstream_zero",
                refresh_policy="per_chunk",
                backend="grouped_fa2",
                system_identity=system_config.identity_dict(),
                method_params=params,
            )
            expected.append(
                {
                    **identity,
                    "profile_config_id": system_id,
                    "method": "transfer_vaware_hybrid_history",
                    "routing_stage": method_spec(
                        "transfer_vaware_hybrid_history"
                    ).routing_stage,
                    "latent_frames": latent_frames,
                    "prompt_id": case["prompt_id"],
                    "seed": case["seed"],
                }
            )
    provenance = {
        "system_holdouts": {
            "artifact_id": holdouts["artifact_id"],
            "sha256": _sha(holdout_path),
            "used_for_profile_content": False,
            "role": "sparse-video authorization gate only",
        },
        "profile_prompts": {
            "artifact_id": prompts["artifact_id"],
            "sha256": _sha(prompt_path),
        },
        "method_params": {"sha256": _sha(method_params_path)},
    }
    suite = {
        "status": "frozen_system_profile_matrix",
        "formal_prompts_used": False,
        "experiment_commit": commit,
        "source_provenance": provenance,
        "methods": ["transfer_vaware_hybrid_history"],
        "method_params": {"transfer_vaware_hybrid_history": params},
        "backend": "grouped_fa2",
        "history_density": 0.25,
        "refresh_policy": "per_chunk",
        "rope_policy": "upstream_zero",
        "cases": cases,
    }
    return suite, {"status": "frozen_system_profile_matrix", "cases": expected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--holdouts", default="configs/formal/system_holdout_prompts.json"
    )
    parser.add_argument(
        "--prompts", default="configs/system/profile_calibration_prompts.json"
    )
    parser.add_argument(
        "--method-params", default="configs/formal/method_params.json"
    )
    parser.add_argument("--commit")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    commit = resolve_experiment_commit(args.commit, repo_root=ROOT)
    suite, expected = build(
        holdout_path=Path(args.holdouts),
        prompt_path=Path(args.prompts),
        method_params_path=Path(args.method_params),
        commit=commit,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "system_profile_suite.json").write_text(
        json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "system_profile_expected.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "pass", "cases": len(suite["cases"])}, indent=2))


if __name__ == "__main__":
    main()
