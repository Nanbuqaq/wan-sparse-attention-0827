#!/usr/bin/env python3
"""Fail-fast public repository size, path, license and secret audit."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


FORBIDDEN_SUFFIXES = {
    ".mp4", ".avi", ".mov", ".pt", ".pth", ".ckpt", ".safetensors",
    ".npy", ".npz", ".log", ".pyc", ".tar", ".zip",
}
FORBIDDEN_PARTS = {"__pycache__", ".pytest_cache", "cache", "outputs", "videos", "logs"}
FORBIDDEN_TEXT = [
    re.compile(r"/home/zhouhe08"),
    re.compile(r"/kaimm-distill"),
    re.compile(r"git\.corp\.kuaishou\.com"),
    re.compile(r"(?:ghp|github_pat)_[A-Za-z0-9_]+"),
    re.compile(r"BEGIN (?:RSA |OPENSSH )?PRIVATE KEY"),
]
REQUIRED = {
    "README.md",
    "SOURCE_LOCK.json",
    "THIRD_PARTY_NOTICES.md",
    "LICENSE",
    ".gitignore",
    "scripts/inferhub_entry.sh",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    parser.add_argument("--output", default="public_audit.json")
    args = parser.parse_args()
    root = Path(args.repo).resolve()
    output_relative = Path(args.output)
    errors = []
    files = []
    listed = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
    ).split(b"\0")
    for raw in listed:
        if not raw:
            continue
        relative = Path(raw.decode("utf-8"))
        path = root / relative
        if "third_party" in relative.parts or not path.is_file():
            continue
        if relative == output_relative:
            continue
        size = path.stat().st_size
        files.append({"path": str(relative), "bytes": size})
        if relative.as_posix() == "scripts/audit_public_repo.py":
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden suffix: {relative}")
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            errors.append(f"forbidden directory: {relative}")
        if size > 2 * 1024 * 1024:
            errors.append(f"file exceeds 2 MiB: {relative} ({size})")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in FORBIDDEN_TEXT:
            if pattern.search(text):
                errors.append(f"forbidden text {pattern.pattern!r}: {relative}")
    missing = sorted(REQUIRED - {item["path"] for item in files})
    errors.extend(f"missing required file: {value}" for value in missing)
    source_lock = json.loads((root / "SOURCE_LOCK.json").read_text(encoding="utf-8"))
    for source in source_lock["sources"]:
        if not source["url"].startswith("https://github.com/"):
            errors.append(f"non-public upstream URL: {source['url']}")
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=root, text=True
    ).strip()
    if branch != "longlive-sparse":
        errors.append(f"unexpected branch: {branch}")
    payload = {
        "status": "pass" if not errors else "fail",
        "branch": branch,
        "files_checked": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "errors": errors,
        "files": files,
    }
    output = root / args.output
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
