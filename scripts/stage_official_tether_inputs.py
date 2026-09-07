#!/usr/bin/env python3
"""Copy public model/dependency inputs into a new, private InferHub bundle.

No existing checkpoint, shared environment, worker, queue, or permissions are
modified. Symlinks in the existing local model bundle are dereferenced.
"""
import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--extra", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination != Path("/kaimm-distill/zhouhe08/longlive-system/tether_official_inputs_v1"):
        raise ValueError("this staging operation is scoped to the dedicated new input bundle")
    if not (args.models / "official_assets_verified.json").is_file():
        raise ValueError("verify official checkpoints before publishing the read-only input bundle")
    destination.mkdir(parents=True, exist_ok=False)
    for name, source in (("models", args.models), ("python-extra", args.extra), ("python-overlay", args.overlay)):
        shutil.copytree(source, destination / name, symlinks=False,
                        ignore=shutil.ignore_patterns(".cache", "__pycache__", "*.pyc", "*.incomplete"))
        print(json.dumps({"copied": name, "destination": str(destination / name)}), flush=True)
    (destination / "staged.ok").write_text("immutable public model/dependency input copy complete\n")


if __name__ == "__main__":
    main()
