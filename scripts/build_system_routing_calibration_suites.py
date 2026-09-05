#!/usr/bin/env python3
"""Build the ten-case isolated motion/state routing calibration."""

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


CALIBRATION_CONFIGS = {
    "rag_dense": {
        "method": "rag_dense",
        "backend": "grouped_fa2",
        "history_density": 1.0,
        "system": LongLiveSystemConfig(),
        "method_params": {},
    },
    "legacy_final": {
        "method": "transfer_vaware_hybrid_history",
        "backend": "grouped_fa2",
        "history_density": 0.25,
        "system": LongLiveSystemConfig(),
        "method_params": {},
    },
    "legacy_final_top_p095": {
        "method": "transfer_vaware_hybrid_history",
        "backend": "grouped_fa2",
        "history_density": 0.25,
        "system": LongLiveSystemConfig(
            group_selection_policy="mass_preserving_top_p",
            group_top_p=0.95,
            group_min_k_ratio=0.10,
        ),
        "method_params": {},
    },
    "system_utility_peak": {
        "method": "system_utility_history",
        "backend": "grouped_fa2",
        "history_density": 0.25,
        "system": LongLiveSystemConfig(),
        "method_params": {
            "value_candidate": "peak_value",
            "cost_strategy": "static_block",
        },
    },
    "system_utility_count_uniform": {
        "method": "system_utility_history",
        "backend": "grouped_fa2",
        "history_density": 0.25,
        "system": LongLiveSystemConfig(),
        "method_params": {
            "value_candidate": "count_uniform",
            "cost_strategy": "static_block",
        },
    },
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    *,
    holdout_path: Path,
    prompt_path: Path,
    method_params_path: Path,
    candidate_path: Path,
    commit: str,
    reuse_dense_motion_commit: str | None = None,
) -> tuple[dict[str, dict], dict]:
    holdouts = load_frozen_system_holdouts(holdout_path)
    prompts = json.loads(prompt_path.read_text(encoding="utf-8"))
    candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
    if prompts.get("formal_prompts_used") is not False or prompts.get("seeds") != [20260904]:
        raise ValueError("routing calibration prompts/seeds are not isolated and frozen")
    if candidates.get("status") != "capture_screened_motion_state_video_calibration_pending":
        raise ValueError("capture-screened routing candidates are not ready")
    all_params = json.loads(method_params_path.read_text(encoding="utf-8"))[
        "method_params"
    ]
    provenance = {
        "holdout_gate": {
            "artifact_id": holdouts["artifact_id"],
            "sha256": _sha(holdout_path),
            "used_for_calibration_content": False,
        },
        "prompts": {"artifact_id": prompts["artifact_id"], "sha256": _sha(prompt_path)},
        "capture_candidates": {
            "artifact_id": candidates["artifact_id"],
            "sha256": _sha(candidate_path),
        },
        "method_params": {"sha256": _sha(method_params_path)},
    }
    suites = {}
    expected = []
    for config_id, frozen in CALIBRATION_CONFIGS.items():
        method = frozen["method"]
        effective_params = {**all_params.get(method, {}), **frozen["method_params"]}
        cases = []
        for prompt in prompts["candidates"]:
            case = {
                "prompt_id": prompt["prompt_id"],
                "category": prompt["category"],
                "prompt": prompt["prompt"],
                "seed": 20260904,
                "latent_frames": 39,
                "history_density": frozen["history_density"],
                "backend": frozen["backend"],
                "refresh_policy": "per_chunk",
                "rope_policy": "upstream_zero",
                "longlive_system": frozen["system"].as_dict(),
                "method_params": frozen["method_params"],
                "calibration_config_id": config_id,
                "record_per_call": True,
                "complete_capture": config_id == "legacy_final",
            }
            cases.append(case)
            identity = build_case_identity(
                commit=(reuse_dense_motion_commit if reuse_dense_motion_commit and
                        config_id == "rag_dense" and case["prompt_id"] == "calibration_motion"
                        else commit),
                method=method,
                prompt_id=case["prompt_id"],
                prompt=case["prompt"],
                seed=case["seed"],
                latent_frames=39,
                history_density=float(case["history_density"]),
                rope_policy="upstream_zero",
                refresh_policy="per_chunk",
                backend=case["backend"],
                system_identity=frozen["system"].identity_dict(),
                method_params=effective_params,
            )
            expected.append(
                {
                    **identity,
                    "calibration_config_id": config_id,
                    "method": method,
                    "routing_stage": method_spec(method).routing_stage,
                    "prompt_id": case["prompt_id"],
                    "seed": 20260904,
                    "latent_frames": 39,
                    "execution_kind": ("external_dense_system_validation" if reuse_dense_motion_commit and
                        config_id == "rag_dense" and case["prompt_id"] == "calibration_motion" else "new_case"),
                }
            )
        params = dict(all_params.get(method, {}))
        params.update(frozen["method_params"])
        suites[config_id] = {
            "status": "frozen_system_routing_calibration_39_latent",
            "formal_prompts_used": False,
            "experiment_commit": commit,
            "calibration_config_id": config_id,
            "source_provenance": provenance,
            "methods": [method],
            "method_params": {method: params},
            "backend": frozen["backend"],
            "history_density": frozen["history_density"],
            "refresh_policy": "per_chunk",
            "rope_policy": "upstream_zero",
            "cases": [case for case in cases if not (reuse_dense_motion_commit and
                      config_id == "rag_dense" and case["prompt_id"] == "calibration_motion")],
        }
    keys = [case["case_key_sha256"] for case in expected]
    if len(set(keys)) != len(keys):
        raise ValueError("routing calibration contains duplicate scientific case identities")
    return suites, {
        "status": "frozen_system_routing_calibration_39_latent",
        "cases": expected,
        "source_provenance": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdouts", default="configs/formal/system_holdout_prompts.json")
    parser.add_argument("--prompts", default="configs/system/profile_calibration_prompts.json")
    parser.add_argument("--method-params", default="configs/formal/method_params.json")
    parser.add_argument("--candidates", default="configs/system/capture_screened_candidates.json")
    parser.add_argument("--commit")
    parser.add_argument("--reuse-dense-motion-commit", help="Reserve one existing Dense validation case; never regenerate it")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    commit = resolve_experiment_commit(args.commit, repo_root=ROOT)
    suites, expected = build(
        holdout_path=Path(args.holdouts),
        prompt_path=Path(args.prompts),
        method_params_path=Path(args.method_params),
        candidate_path=Path(args.candidates),
        commit=commit,
        reuse_dense_motion_commit=args.reuse_dense_motion_commit,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for config_id, suite in suites.items():
        (output / f"suite_{config_id}.json").write_text(
            json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (output / "expected_protocol.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    new_expected = {**expected, 'cases': [case for case in expected['cases'] if case['execution_kind'] == 'new_case']}
    (output / "expected.json").write_text(json.dumps(new_expected, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "pass", "configs": 5, "protocol_cases": 10,
                      "new_cases": len(new_expected['cases']), "external_cases_pending_audit": 10-len(new_expected['cases'])}, indent=2))


if __name__ == "__main__":
    main()
