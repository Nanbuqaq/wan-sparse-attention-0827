#!/usr/bin/env python3
"""Ensure every expected GPU case reaches pass/fail/negative with valid evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import av
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.longlive_sparse.case_identity import validate_case_identity


TERMINAL = {"pass", "fail", "negative"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decoded_frames(path: Path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", required=True)
    parser.add_argument("--states", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-artifacts", action="store_true")
    args = parser.parse_args()
    expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))["cases"]
    states = json.loads(Path(args.states).read_text(encoding="utf-8"))["cases"]
    by_id = {}
    errors = []
    for state in states:
        case_id = state.get("id", state.get("case_id"))
        if not case_id:
            errors.append("state without id")
            continue
        if case_id in by_id:
            errors.append(f"duplicate terminal state: {case_id}")
            continue
        by_id[case_id] = state
    records = []
    for case in expected:
        identity_errors = validate_case_identity(case)
        errors.extend(f"invalid expected identity {case['id']}: {error}" for error in identity_errors)
        state = by_id.get(case["id"])
        if state is None:
            errors.append(f"missing terminal state: {case['id']}")
            continue
        state_identity_errors = validate_case_identity(state)
        errors.extend(f"invalid state identity {case['id']}: {error}" for error in state_identity_errors)
        if state.get("case_key_sha256") != case.get("case_key_sha256"):
            errors.append(f"state/expected case identity mismatch: {case['id']}")
        status = state.get("status")
        if status not in TERMINAL:
            errors.append(f"non-terminal state {status!r}: {case['id']}")
        if status in {"pass", "negative"}:
            required = ["backend", "stats", "config"]
            if state.get("routing_stage") != "N/A":
                required.append("route_plan_sha256")
            for key in required:
                if not state.get(key):
                    errors.append(f"successful case missing {key}: {case['id']}")
            if state.get("failed_calls", 0) or state.get("fallback_calls", 0) or state.get("nan_calls", 0):
                errors.append(f"successful case has failed/fallback/NaN calls: {case['id']}")
            if state.get("backend") in {"unknown", "fallback", "dense_fallback"}:
                errors.append(f"successful case did not hit a verified backend: {case['id']}")
            if state.get("backend") != case.get("case_key", {}).get("backend", case.get("backend")):
                errors.append(f"declared backend differs from frozen case key: {case['id']}")
            stats_path = Path(str(state.get("stats", "")))
            config_path = Path(str(state.get("config", "")))
            if not stats_path.is_file() or not config_path.is_file():
                errors.append(f"successful case missing stats/config artifact: {case['id']}")
            else:
                stats = json.loads(stats_path.read_text(encoding="utf-8"))
                for key in ("history_density", "global_executed_density"):
                    value = stats.get(key)
                    if value is None or not 0.0 <= float(value) <= 1.0:
                        errors.append(f"invalid {key} denominator/value: {case['id']}")
                transfer_density = stats.get("history_transfer_density")
                if transfer_density is not None and not 0.0 <= float(transfer_density) <= 1.0:
                    errors.append(f"invalid history_transfer_density: {case['id']}")
                if stats.get("failed_calls", 0) or stats.get("dense_fallback_calls", 0):
                    errors.append(f"stats contain failed/fallback calls: {case['id']}")
            if status == "negative" and not state.get("negative_reasons"):
                errors.append(f"negative case missing reasons: {case['id']}")
            if not args.skip_artifacts:
                video_value = str(state.get("video", ""))
                video = Path(video_value) if video_value else Path("missing-video")
                latent_value = state.get("latent")
                latent = (
                    Path(str(latent_value))
                    if latent_value
                    else video.parent / "latents.pt"
                )
                if not video.is_file() or not latent.is_file():
                    errors.append(f"successful case missing video/latent artifact: {case['id']}")
                else:
                    latent_frames = int(case["latent_frames"])
                    decoded = _decoded_frames(video)
                    if decoded != 4 * latent_frames - 3:
                        errors.append(f"decoded frame count {decoded} is incomplete: {case['id']}")
                    tensor = torch.load(latent, map_location="cpu", weights_only=True)
                    expected_shape = (1, latent_frames, 16, 60, 104)
                    if tuple(tensor.shape) != expected_shape:
                        errors.append(f"latent shape {tuple(tensor.shape)} != {expected_shape}: {case['id']}")
                    if not torch.isfinite(tensor).all():
                        errors.append(f"non-finite latent: {case['id']}")
                    if state.get("video_sha256") and _sha256(video) != state["video_sha256"]:
                        errors.append(f"video SHA mismatch: {case['id']}")
        if status == "fail" and not state.get("failure_reason"):
            errors.append(f"failed case missing reason: {case['id']}")
        records.append(state)
    payload = {
        "status": "pass" if not errors else "fail",
        "expected_cases": len(expected),
        "terminal_cases": len(records),
        "pass_cases": sum(case.get("status") == "pass" for case in records),
        "negative_cases": sum(case.get("status") == "negative" for case in records),
        "fail_cases": sum(case.get("status") == "fail" for case in records),
        "errors": errors,
        "cases": records,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
