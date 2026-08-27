"""Public-safe model path resolution."""

from __future__ import annotations

import os
from pathlib import Path


def wan_model_path() -> Path:
    value = os.environ.get("WAN_MODEL_PATH")
    if not value:
        raise RuntimeError("WAN_MODEL_PATH must point to a local Wan Diffusers model")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path

