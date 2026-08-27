"""Strict loaders for the pinned upstream routing and reviewed fixed kernel."""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENDORED_SVOO_ROOT = ROOT / "adapters" / "vendor" / "svoo_repo"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_svoo_core():
    required = VENDORED_SVOO_ROOT / "svoo" / "co_clustering.py"
    if not required.is_file():
        raise FileNotFoundError(required)
    sys.path.insert(0, str(VENDORED_SVOO_ROOT))
    module = importlib.import_module("svoo.co_clustering")
    if not Path(module.__file__).resolve().is_relative_to(VENDORED_SVOO_ROOT.resolve()):
        raise RuntimeError(f"SVOO core imported from unexpected location: {module.__file__}")
    return module


def load_svoo_permutation():
    sys.path.insert(0, str(VENDORED_SVOO_ROOT))
    module = importlib.import_module("svoo.kernels.triton.permute")
    if not Path(module.__file__).resolve().is_relative_to(VENDORED_SVOO_ROOT.resolve()):
        raise RuntimeError(f"SVOO permutation imported from unexpected location: {module.__file__}")
    return module


def source_hashes() -> dict[str, str]:
    paths = {
        "vendored_svoo_co_clustering": VENDORED_SVOO_ROOT / "svoo" / "co_clustering.py",
        "vendored_svoo_permutation": VENDORED_SVOO_ROOT
        / "svoo"
        / "kernels"
        / "triton"
        / "permute.py",
        "cleanroom_fixed64_kernel": ROOT / "adapters" / "kernels_fixed64.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}
