"""Pinned, read-only LongLive upstream loading utilities."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LONGLIVE_BASE = Path(
    os.environ.get(
        "LONGLIVE_BASE_SOURCE",
        str(PROJECT_ROOT / "third_party/longlive-inferhub"),
    )
).resolve()
LONGLIVE_BASE_COMMIT = "fc494740e9bf8c6bc9d0f3cbe01e68c7a2fd9fc7"
LONGLIVE_RAG = Path(
    os.environ.get(
        "LONGLIVE_RAG_SOURCE",
        str(PROJECT_ROOT / "third_party/LongLive-RAG"),
    )
).resolve()
LONGLIVE_RAG_COMMIT = "973884a3cd3ad4b314c3d4ab42274c52e7a0b22a"


def configure_upstream_paths() -> None:
    """Put the validated LongLive base before the read-only RAG repository."""

    for path in (str(LONGLIVE_RAG), str(LONGLIVE_BASE)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    # The second insertion leaves the validated base first.


def load_module_from_path(name: str, path: Path) -> ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_latentmem_module() -> ModuleType:
    configure_upstream_paths()
    return load_module_from_path(
        "longlive_sparse_upstream_latentmem",
        LONGLIVE_RAG / "wan/modules/causal_model_latentmem.py",
    )


def load_rag_pipeline_module() -> ModuleType:
    configure_upstream_paths()
    return load_module_from_path(
        "longlive_sparse_upstream_rag_pipeline",
        LONGLIVE_RAG / "pipeline/causal_inference.py",
    )


def provenance() -> dict[str, str]:
    return {
        "longlive_base_path": str(LONGLIVE_BASE),
        "longlive_base_commit": LONGLIVE_BASE_COMMIT,
        "longlive_rag_path": str(LONGLIVE_RAG),
        "longlive_rag_commit": LONGLIVE_RAG_COMMIT,
    }
