"""Runtime path and cache setup confined to the short-video workstream."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def configure_runtime() -> None:
    overlays = [ROOT / ".runtime" / "python"]
    overlay_file = ROOT / ".runtime" / "python_overlays.txt"
    if overlay_file.is_file():
        overlays.extend(
            Path(line.strip())
            for line in overlay_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    overlays.extend(
        Path(value)
        for value in os.environ.get("WAN_SPARSE_PYTHON_OVERLAYS", "").split(os.pathsep)
        if value
    )
    for path in overlays:
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    cache = ROOT / ".runtime" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    defaults = {
        "XDG_CACHE_HOME": cache / "xdg",
        "HF_HOME": cache / "huggingface",
        "TORCH_HOME": cache / "torch",
        "TRITON_CACHE_DIR": cache / "triton",
    }
    for name, value in defaults.items():
        value = Path(value)
        value.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(value))
