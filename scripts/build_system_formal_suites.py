#!/usr/bin/env python3
"""Build four isolated one-method suites for conditional 477/957 expansion."""

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
from adapters.longlive_sparse.system_formal import validate_system_method_freeze


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    *,
    holdout_path: Path,
    method_freeze_path: Path,
    method_params_path: Path,
    latent_frames: int,
    commit: str,
) -> tuple[dict[str, dict], dict]:
    if latent_frames not in {120, 240}:
        raise ValueError("formal system suites support only 120 or 240 latent frames")
    holdouts = load_frozen_system_holdouts(holdout_path)
    method_freeze = json.loads(method_freeze_path.read_text(encoding="utf-8"))
    errors = validate_system_method_freeze(method_freeze)
    if errors:
        raise ValueError("invalid system method freeze: " + "; ".join(errors))
    all_method_params = json.loads(method_params_path.read_text(encoding="utf-8"))[
        "method_params"
    ]
    seeds = (
        holdouts["formal_477_seeds"]
        if latent_frames == 120
        else holdouts["formal_957_seeds"]
    )
    provenance = {
        "holdouts": {"artifact_id": holdouts["artifact_id"], "sha256": _sha(holdout_path)},
        "method_freeze": {
            "artifact_id": method_freeze["artifact_id"],
            "sha256": _sha(method_freeze_path),
        },
        "method_params": {"sha256": _sha(method_params_path)},
    }
    suites = {}
    expected = []
    for frozen in method_freeze["configs"]:
        method = frozen["method"]
        effective_params = {**all_method_params.get(method, {}), **frozen.get("method_params", {})}
        cases = []
        for prompt in holdouts["prompts"]:
            for seed in seeds:
                case = {
                    "prompt_id": prompt["prompt_id"],
                    "category": prompt["category"],
                    "prompt": prompt["prompt"],
                    "seed": int(seed),
                    "latent_frames": latent_frames,
                    "backend": frozen["backend"],
                    "history_density": frozen["history_density"],
                    "refresh_policy": frozen.get("refresh_policy", "per_chunk"),
                    "rope_policy": frozen.get("rope_policy", "upstream_zero"),
                    "longlive_system": LongLiveSystemConfig.from_mapping(frozen.get("longlive_system")).identity_dict(),
                    "method_params": frozen.get("method_params", {}),
                    "formal_config_id": frozen["config_id"],
                }
                cases.append(case)
                identity = build_case_identity(
                    commit=commit,
                    method=method,
                    prompt_id=case["prompt_id"],
                    prompt=case["prompt"],
                    seed=case["seed"],
                    latent_frames=latent_frames,
                    history_density=float(case["history_density"]),
                    rope_policy=case["rope_policy"],
                    refresh_policy=case["refresh_policy"],
                    backend=case["backend"],
                    system_identity=case["longlive_system"],
                    method_params=effective_params,
                )
                expected.append(
                    {
                        **identity,
                        "formal_config_id": frozen["config_id"],
                        "method": method,
                        "routing_stage": method_spec(method).routing_stage,
                        "prompt_id": case["prompt_id"],
                        "seed": case["seed"],
                        "latent_frames": latent_frames,
                    }
                )
        params = dict(all_method_params.get(method, {}))
        params.update(frozen.get("method_params", {}))
        suites[frozen["config_id"]] = {
            "status": f"frozen_system_formal_{4 * latent_frames - 3}",
            "formal_prompts_used": True,
            "experiment_commit": commit,
            "formal_config_id": frozen["config_id"],
            "source_provenance": provenance,
            "methods": [method],
            "method_params": {method: params},
            "backend": frozen["backend"],
            "history_density": frozen["history_density"],
            "refresh_policy": frozen.get("refresh_policy", "per_chunk"),
            "rope_policy": frozen.get("rope_policy", "upstream_zero"),
            "cases": cases,
        }
    return suites, {
        "status": f"frozen_system_formal_{4 * latent_frames - 3}",
        "cases": expected,
        "source_provenance": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdouts", default="configs/formal/system_holdout_prompts.json")
    parser.add_argument("--method-freeze", default="configs/formal/system_method_freeze.json")
    parser.add_argument("--method-params", default="configs/formal/method_params.json")
    parser.add_argument("--latent-frames", type=int, choices=(120, 240), required=True)
    parser.add_argument("--commit")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    commit = resolve_experiment_commit(args.commit, repo_root=ROOT)
    suites, expected = build(
        holdout_path=Path(args.holdouts),
        method_freeze_path=Path(args.method_freeze),
        method_params_path=Path(args.method_params),
        latent_frames=args.latent_frames,
        commit=commit,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for config_id, suite in suites.items():
        (output / f"suite_{config_id}.json").write_text(
            json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (output / "expected.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "pass", "configs": len(suites), "cases": len(expected["cases"])}, indent=2))


if __name__ == "__main__":
    main()
