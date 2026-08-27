#!/usr/bin/env python3
"""Build a SHA-256 manifest for source, configs, metrics, figures, videos, and logs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/manifests/SHA256SUMS.json")
    args = parser.parse_args()
    output = ROOT / args.output
    roots = ["adapters", "configs", "docs", "scripts", "tests", "results/metrics", "results/figures", "results/videos", "results/logs", "results/manifests"]
    files = [ROOT / "README.md", ROOT / "AGENT_PROMPT.md"]
    for relative in roots:
        base = ROOT / relative
        if base.is_dir():
            files.extend(path for path in base.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    entries = [
        {
            "path": str(path.relative_to(ROOT)),
            "size": path.stat().st_size,
            "sha256": digest(path),
        }
        for path in sorted(set(files))
        if path.resolve()
        not in {
            output.resolve(),
            output.with_name("SHA256SUMS.txt").resolve(),
            (ROOT / "results/logs/manifest_check.log").resolve(),
        }
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"algorithm": "sha256", "files": entries}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    text_path = (
        output.with_name("SHA256SUMS.txt")
        if output.name == "SHA256SUMS.json"
        else output.with_suffix(".txt")
    )
    text_path.write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in entries), encoding="utf-8"
    )
    print(f"[saved] {output} files={len(entries)}")


if __name__ == "__main__":
    main()
