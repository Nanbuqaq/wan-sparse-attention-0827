#!/usr/bin/env python3
"""Run pinned LongLive inference with the local sparse-history pipeline proxy."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE = Path(
    os.environ.get(
        "LONGLIVE_BASE_SOURCE",
        str(ROOT / "third_party/longlive-inferhub"),
    )
).resolve()
OVERLAY = Path(os.environ.get("LONGLIVE_PYTHON_OVERLAY", ""))


def _prepend(path: Path) -> None:
    value = str(path)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()
    config_path = Path(args.config_path).resolve()
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runtime_mode = raw_config.get("runtime_mode", "rag_sparse")
    if runtime_mode not in {"native_dense", "rag_sparse"}:
        raise ValueError(f"unsupported runtime_mode: {runtime_mode!r}")

    if str(OVERLAY) and OVERLAY.is_dir():
        _prepend(OVERLAY)
    _prepend(BASE)
    _prepend(ROOT)
    os.environ.setdefault("LONGLIVE_DISABLE_FA3", "1")
    os.environ.setdefault(
        "LONGLIVE_WAN_MODELS_ROOT",
        str(ROOT / "models"),
    )

    if runtime_mode == "rag_sparse":
        import pipeline
        from adapters.longlive_sparse.runtime import build_sparse_pipeline

        class PipelineProxy:
            def __new__(cls, pipeline_args, device, *unused_args, **unused_kwargs):
                del cls, unused_args, unused_kwargs
                return build_sparse_pipeline(pipeline_args, device)

        pipeline.CausalInferencePipeline = PipelineProxy

    previous_argv = sys.argv[:]
    sys.argv = [str(BASE / "inference.py"), "--config_path", str(config_path)]
    try:
        namespace = runpy.run_path(str(BASE / "inference.py"), run_name="__main__")
    finally:
        sys.argv = previous_argv

    pipeline_object = namespace.get("pipeline")
    config = namespace.get("config")
    if pipeline_object is None or config is None:
        raise RuntimeError("upstream inference did not expose pipeline/config state")
    output_dir = Path(str(config.output_folder)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    latents = namespace.get("latents")
    if isinstance(latents, torch.Tensor):
        torch.save(latents.detach().cpu(), output_dir / "latents.pt")
    if hasattr(pipeline_object, "sparse_history_archive"):
        payload = pipeline_object.sparse_history_aggregate_stats.as_dict()
        payload["config"] = pipeline_object.sparse_history_config.as_dict()
        payload["runs"] = pipeline_object.sparse_history_completed_runs
    else:
        payload = {
            "method": "native_dense",
            "attention_backend": namespace.get("backend", "unknown"),
            "calls": 0,
            "history_density": 1.0,
            "global_executed_density": 1.0,
        }
    (output_dir / "sparse_history_stats.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("SPARSE_HISTORY_STATS " + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
