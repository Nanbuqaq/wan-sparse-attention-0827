#!/usr/bin/env python3
"""Run matched formal-basic video quality groups on one or more local GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_case_metrics import baseline_method


REVIEWABLE = {"pass", "negative"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pairing_key(case: dict) -> tuple:
    return (
        case.get("commit"),
        case.get("prompt_id"),
        int(case.get("seed")),
        int(case.get("latent_frames")),
    )


def build_quality_groups(cases: list[dict]) -> tuple[list[dict], list[str]]:
    baselines = {}
    for case in cases:
        if case.get("method") not in {"native_dense", "rag_dense"}:
            continue
        token = (case["method"], *pairing_key(case))
        if token in baselines:
            raise RuntimeError(f"duplicate Dense baseline: {token}")
        baselines[token] = case

    groups = {}
    skipped = []
    for case in cases:
        if case.get("status") not in REVIEWABLE:
            skipped.append(str(case.get("id", case.get("case_id"))))
            continue
        method = str(case["method"])
        baseline_name = baseline_method(method)
        token = (baseline_name, *pairing_key(case))
        baseline = baselines.get(token)
        if baseline is None or baseline.get("status") not in REVIEWABLE:
            raise RuntimeError(f"missing reviewable matched baseline: {case['id']}")
        if not Path(str(case.get("video", ""))).is_file():
            raise FileNotFoundError(f"quality video missing: {case['id']}")
        group_id = (
            f"{baseline_name}__{case['prompt_id']}__s{int(case['seed'])}"
            f"__lf{int(case['latent_frames'])}"
        )
        group = groups.setdefault(
            group_id,
            {
                "group_id": group_id,
                "baseline": baseline,
                "cases": {},
            },
        )
        if method in group["cases"]:
            raise RuntimeError(f"duplicate method in quality group: {group_id} {method}")
        group["cases"][method] = case

    output = []
    for group_id, group in groups.items():
        baseline = group["baseline"]
        group["cases"].setdefault(baseline["method"], baseline)
        output.append(
            {
                "group_id": group_id,
                "baseline": baseline,
                "cases": [
                    group["cases"][method] for method in sorted(group["cases"])
                ],
            }
        )
    output.sort(key=lambda item: item["group_id"])
    return output, sorted(skipped)


def assign_groups(groups: list[dict], devices: list[str | None]) -> list[list[dict]]:
    lanes = [[] for _ in devices]
    loads = [0 for _ in devices]
    for group in sorted(groups, key=lambda item: (-len(item["cases"]), item["group_id"])):
        lane = min(range(len(devices)), key=lambda index: (loads[index], index))
        lanes[lane].append(group)
        loads[lane] += len(group["cases"])
    return lanes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--lpips-config", default="configs/quality/lpips_alex_v0p1.json"
    )
    parser.add_argument("--lpips-weights", required=True)
    parser.add_argument("--lpips-trunk-weights", required=True)
    parser.add_argument(
        "--devices",
        default="",
        help="comma-separated physical GPU ids; empty runs one CPU lane",
    )
    args = parser.parse_args()
    states_path = Path(args.states).resolve()
    states = json.loads(states_path.read_text(encoding="utf-8"))["cases"]
    groups, skipped = build_quality_groups(states)
    if not groups:
        raise RuntimeError("no reviewable formal quality group")

    config_path = Path(args.lpips_config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_before_formal_quality":
        raise ValueError("LPIPS quality configuration is not frozen")
    linear = config["linear_weights"]
    trunk = config["trunk_weights"]
    versions = config["package_versions"]
    weights_path = Path(args.lpips_weights).resolve()
    trunk_path = Path(args.lpips_trunk_weights).resolve()

    device_values = [value.strip() for value in args.devices.split(",") if value.strip()]
    if len(device_values) != len(set(device_values)):
        raise ValueError("--devices contains duplicates")
    devices: list[str | None] = device_values or [None]
    lanes = assign_groups(groups, devices)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def run_lane(lane_index: int) -> list[dict]:
        device = devices[lane_index]
        lane_results = []
        for group in lanes[lane_index]:
            group_output = output_dir / group["group_id"]
            group_output.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_videos.py"),
                "--reference",
                str(group["baseline"]["video"]),
                "--output-dir",
                str(group_output),
                "--lpips-weights",
                str(weights_path),
                "--lpips-weights-sha256",
                linear["sha256"],
                "--lpips-package-version",
                versions["lpips"],
                "--lpips-trunk-weights",
                str(trunk_path),
                "--lpips-trunk-weights-sha256",
                trunk["sha256"],
                "--torch-package-version",
                versions["torch"],
                "--torchvision-package-version",
                versions["torchvision"],
            ]
            expected_ids = []
            for case in group["cases"]:
                case_id = str(case.get("id", case.get("case_id")))
                expected_ids.append(case_id)
                command.extend(["--candidate", f"{case_id}={case['video']}"])
            environment = os.environ.copy()
            if device is not None:
                environment["CUDA_VISIBLE_DEVICES"] = device
            log_path = group_output / "quality.log"
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            summary_path = group_output / "paired_video_summary.json"
            errors = []
            if completed.returncode:
                errors.append(f"evaluate_videos exit {completed.returncode}")
            elif not summary_path.is_file():
                errors.append("paired_video_summary.json missing")
            else:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                observed_ids = sorted(summary.get("candidates", {}))
                if observed_ids != sorted(expected_ids):
                    errors.append("quality summary candidate set mismatch")
            lane_results.append(
                {
                    "group_id": group["group_id"],
                    "device": device if device is not None else "cpu",
                    "baseline_id": group["baseline"]["id"],
                    "candidate_ids": expected_ids,
                    "status": "pass" if not errors else "fail",
                    "errors": errors,
                    "summary": str(summary_path),
                    "log": str(log_path),
                }
            )
        return lane_results

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(run_lane, index) for index in range(len(devices))]
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())
    results.sort(key=lambda item: item["group_id"])
    errors = [
        f"{item['group_id']}: {error}"
        for item in results
        for error in item["errors"]
    ]
    manifest = {
        "status": "pass" if not errors else "fail",
        "states": {"path": str(states_path), "sha256": sha256(states_path)},
        "lpips_config": {
            "path": str(config_path.resolve()),
            "sha256": sha256(config_path),
        },
        "devices": [value if value is not None else "cpu" for value in devices],
        "groups": results,
        "skipped_nonreviewable_cases": skipped,
        "errors": errors,
        "statistical_unit": "complete video",
    }
    manifest_path = output_dir / "quality_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "groups": len(results),
                "reviewable_cases": sum(len(item["candidate_ids"]) for item in results),
                "skipped": len(skipped),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
