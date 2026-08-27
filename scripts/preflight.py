#!/usr/bin/env python3
"""Record a read-only dependency, source, model, and GPU preflight."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

from bootstrap import ROOT, configure_runtime
from model_path import wan_model_path

configure_runtime()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from adapters.vendor import source_hashes


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    import diffusers
    import triton

    model_config = wan_model_path() / "transformer" / "config.json"
    payload = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "diffusers": diffusers.__version__,
        "triton": triton.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ],
        "model_config": str(model_config),
        "model_config_sha256": digest(model_config),
        "source_hashes": source_hashes(),
        "status": "pass" if torch.cuda.is_available() else "fail",
    }
    output = ROOT / "results" / "manifests" / "preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
