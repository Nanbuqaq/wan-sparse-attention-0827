#!/usr/bin/env python3
"""Create a reusable content-hash identity for local Wan model files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bootstrap import ROOT, configure_runtime
from model_path import wan_model_path

configure_runtime()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.dependencies import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/manifests/model_identity.json")
    args = parser.parse_args()
    model = wan_model_path()
    files = sorted(
        path
        for path in model.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".json", ".txt", ".model", ".safetensors"}
    )
    entries = []
    for index, path in enumerate(files):
        print(f"[hash] {index + 1}/{len(files)} {path.relative_to(model)}", flush=True)
        entries.append(
            {
                "path": str(path.relative_to(model)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "schema_version": 1,
        "model_root_name": model.name,
        "files": entries,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "files": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
