#!/usr/bin/env python3
"""Create immutable source-only vendor archives for the official reproduction.

Source archives are code from the submitted Git commit, not mutable code read
from a weights directory. Upstream licenses and per-file hashes are retained.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


EXCLUDED = {".git", "__pycache__", ".pytest_cache", "node_modules", "assets", "outputs", "notebooks", "demo", "data", "images", "videos"}
SUFFIXES = {".py", ".yaml", ".yml", ".json", ".toml", ".cfg", ".txt", ".md", ".cu", ".cuh", ".cpp", ".h", ".sh"}


def package(source, output, revision, url):
    source = source.resolve()
    files = [p for p in sorted(source.rglob("*")) if p.is_file() and not p.is_symlink()
             and not set(p.relative_to(source).parts).intersection(EXCLUDED)
             and (p.suffix in SUFFIXES or p.name.startswith("LICENSE"))]
    entries = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for path in files:
                content = path.read_bytes()
                relative = path.relative_to(source).as_posix()
                info = tarfile.TarInfo(relative)
                info.size, info.mode, info.mtime = len(content), 0o644, 0
                archive.addfile(info, io.BytesIO(content))
                entries.append({"path": relative, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    lock = {"source": url, "revision": revision, "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "files": entries, "unchanged_source_files": True, "excluded_nonruntime_assets": sorted(EXCLUDED)}
    with output.with_suffix(".lock.json").open("x") as handle:
        handle.write(json.dumps(lock, indent=2) + "\n")
    print(json.dumps({"archive": str(output), "files": len(entries), "bytes": output.stat().st_size}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    package(args.source, args.output, args.revision, args.url)


if __name__ == "__main__":
    main()
