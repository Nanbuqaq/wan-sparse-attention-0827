#!/usr/bin/env python3
"""Recover a terminal formal-basic InferHub batch into the outer results tree."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {"pass", "fail", "negative"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_bucket(relative: Path) -> str:
    if relative.name == "video.mp4":
        return "videos"
    if relative.name == "latents.pt":
        return "latents"
    if "qkv_captures" in relative.parts and relative.suffix == ".pt":
        return "captures"
    if relative.suffix == ".log":
        return "logs"
    if relative.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return "reviews"
    return "manifests"


def result_roots(results_root: Path, *, run_id: str, date_tag: str) -> dict[str, Path]:
    return {
        "manifests": results_root / "manifests" / run_id,
        "logs": results_root / "logs" / f"inferhub_{date_tag}" / run_id,
        "videos": results_root / "videos" / f"gpu_{date_tag}" / run_id,
        "latents": results_root / "latents" / f"gpu_{date_tag}" / run_id,
        "captures": results_root / "captures" / f"gpu_{date_tag}" / run_id,
        "reviews": results_root / "reviews" / run_id,
        "audits": results_root / "audits" / run_id,
    }


def read_sha_manifest(path: Path) -> list[tuple[str, Path]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, value = line.split(None, 1)
        value = value.strip()
        if value.startswith("*"):
            value = value[1:]
        records.append((digest, Path(value)))
    return records


def validate_terminal_source(
    source_root: Path,
    *,
    expected_cases: int,
    expected_relative: Path = Path("control/expected_basic_477.json"),
) -> dict:
    required = {
        "batch_status": source_root / "batch_status.json",
        "merged_states": source_root / "merged_case_states.json",
        "terminal_audit": source_root / "terminal_state_audit.json",
        "sha_manifest": source_root / "SHA256SUMS.txt",
        "expected": source_root / expected_relative,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"terminal batch artifacts missing: {missing}")

    batch = json.loads(required["batch_status"].read_text(encoding="utf-8"))
    merged = json.loads(required["merged_states"].read_text(encoding="utf-8"))
    audit = json.loads(required["terminal_audit"].read_text(encoding="utf-8"))
    expected = json.loads(required["expected"].read_text(encoding="utf-8"))
    cases = merged["cases"]
    statuses = Counter(case.get("status") for case in cases)
    errors = []
    if batch.get("terminal_audit_completed") is not True:
        errors.append("batch did not complete terminal audit")
    if audit.get("status") != "pass" or audit.get("errors"):
        errors.append("source terminal audit is not pass")
    if len(expected.get("cases", [])) != expected_cases:
        errors.append("expected manifest case count mismatch")
    if len(cases) != expected_cases or len({case.get("id") for case in cases}) != expected_cases:
        errors.append("merged terminal case count/identity mismatch")
    if int(audit.get("expected_cases", -1)) != expected_cases:
        errors.append("terminal audit expected case count mismatch")
    if int(audit.get("terminal_cases", -1)) != expected_cases:
        errors.append("terminal audit terminal case count mismatch")
    invalid = sorted(set(statuses) - TERMINAL, key=str)
    if invalid:
        errors.append(f"non-terminal statuses: {invalid}")
    sha_records = read_sha_manifest(required["sha_manifest"])
    successful = statuses["pass"] + statuses["negative"]
    if len(sha_records) != 2 * successful:
        errors.append(
            f"source SHA manifest has {len(sha_records)} entries for {successful} successful cases"
        )
    sha_errors = []
    for digest, path in sha_records:
        if not path.is_file():
            sha_errors.append(f"missing SHA artifact: {path}")
        elif sha256(path) != digest:
            sha_errors.append(f"SHA mismatch: {path}")
    errors.extend(sha_errors)
    if errors:
        raise RuntimeError("; ".join(errors))
    return {
        "expected_cases": expected_cases,
        "statuses": dict(statuses),
        "lane_statuses": batch.get("lane_statuses"),
        "source_terminal_audit_sha256": sha256(required["terminal_audit"]),
        "source_merged_states_sha256": sha256(required["merged_states"]),
        "source_sha_manifest_sha256": sha256(required["sha_manifest"]),
        "source_sha_entries": len(sha_records),
    }


def copy_verified(
    source_root: Path,
    source: Path,
    roots: dict[str, Path],
) -> dict:
    relative = source.relative_to(source_root)
    bucket = artifact_bucket(relative)
    destination = roots[bucket] / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_sha = sha256(source)
    shutil.copy2(source, destination)
    destination_sha = sha256(destination)
    if destination_sha != source_sha:
        raise RuntimeError(f"recovered SHA mismatch: {relative}")
    return {
        "relative_path": str(relative),
        "bucket": bucket,
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "bytes": source.stat().st_size,
        "sha256": source_sha,
    }


def mapped_path(
    value: str | None,
    *,
    source_root: Path,
    roots: dict[str, Path],
) -> str | None:
    if not value:
        return value
    source = Path(value)
    try:
        relative = source.resolve().relative_to(source_root.resolve())
    except ValueError:
        return value
    return str((roots[artifact_bucket(relative)] / relative).resolve())


def rewrite_local_states(
    payload: dict,
    *,
    source_root: Path,
    roots: dict[str, Path],
) -> dict:
    cases = []
    for original in payload["cases"]:
        case = dict(original)
        case["video"] = mapped_path(
            case.get("video"), source_root=source_root, roots=roots
        )
        case["stats"] = mapped_path(
            case.get("stats"), source_root=source_root, roots=roots
        )
        case["config"] = mapped_path(
            case.get("config"), source_root=source_root, roots=roots
        )
        if case.get("status") in {"pass", "negative"}:
            source_video = Path(str(original["video"]))
            source_latent = Path(str(original.get("latent", source_video.parent / "latents.pt")))
            case["latent"] = mapped_path(
                str(source_latent), source_root=source_root, roots=roots
            )
        case["recovered_from"] = str(source_root.resolve())
        cases.append(case)
    return {**payload, "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--date-tag", required=True)
    parser.add_argument("--expected-cases", type=int, default=44)
    parser.add_argument(
        "--expected-relative", default="control/expected_basic_477.json"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--extra-log", action="append", default=[])
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    source_root = Path(args.source_root).resolve()
    results_root = Path(args.results_root).resolve()
    expected_relative = Path(args.expected_relative)
    if expected_relative.is_absolute() or ".." in expected_relative.parts:
        raise ValueError("--expected-relative must stay inside the batch root")
    source_summary = validate_terminal_source(
        source_root,
        expected_cases=args.expected_cases,
        expected_relative=expected_relative,
    )
    roots = result_roots(results_root, run_id=args.run_id, date_tag=args.date_tag)
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)

    source_files = sorted(path for path in source_root.rglob("*") if path.is_file())
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(copy_verified, source_root, path, roots)
            for path in source_files
        ]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda item: item["relative_path"])

    for value in args.extra_log:
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"extra log missing: {path}")
        destination = roots["logs"] / "platform" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        if sha256(path) != sha256(destination):
            raise RuntimeError(f"extra log SHA mismatch: {path}")

    merged_source = json.loads(
        (source_root / "merged_case_states.json").read_text(encoding="utf-8")
    )
    local_states = rewrite_local_states(
        merged_source, source_root=source_root, roots=roots
    )
    local_states_path = roots["manifests"] / "merged_case_states.local.json"
    local_states_path.write_text(
        json.dumps(local_states, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    local_audit_path = roots["audits"] / "terminal_state_audit.local.json"
    local_audit_log = roots["logs"] / "local_terminal_audit.log"
    audit = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_case_states.py"),
            "--expected",
            str(roots["manifests"] / expected_relative),
            "--states",
            str(local_states_path),
            "--output",
            str(local_audit_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    local_audit_log.write_text(audit.stdout, encoding="utf-8")
    if audit.returncode:
        raise RuntimeError(
            f"local recovered artifact audit failed; see {local_audit_log}"
        )

    bucket_counts = Counter(record["bucket"] for record in records)
    bucket_bytes = Counter()
    for record in records:
        bucket_bytes[record["bucket"]] += int(record["bytes"])
    recovery = {
        "status": "pass",
        "source_root": str(source_root),
        "results_root": str(results_root),
        "run_id": args.run_id,
        "date_tag": args.date_tag,
        "expected_relative": str(expected_relative),
        "source_summary": source_summary,
        "bucket_roots": {name: str(path.resolve()) for name, path in roots.items()},
        "bucket_counts": dict(bucket_counts),
        "bucket_bytes": dict(bucket_bytes),
        "copied_files": len(records),
        "copied_bytes": sum(int(record["bytes"]) for record in records),
        "local_states": str(local_states_path),
        "local_states_sha256": sha256(local_states_path),
        "local_terminal_audit": str(local_audit_path),
        "local_terminal_audit_sha256": sha256(local_audit_path),
        "records": records,
    }
    recovery_path = roots["audits"] / "recovery_audit.json"
    recovery_path.write_text(
        json.dumps(recovery, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "files": len(records),
                "bytes": recovery["copied_bytes"],
                "statuses": source_summary["statuses"],
                "recovery_audit": str(recovery_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
