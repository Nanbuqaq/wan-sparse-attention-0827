#!/usr/bin/env python3
"""Load LongLive-RAG once, then run a shard of distinct method/video cases."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import torch
import yaml
from einops import rearrange
from torchvision.io import write_video


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _route_digest(stats: dict) -> tuple[str | None, list[str]]:
    values = sorted(stats.get("route_plan_sha256_counts", {}))
    if not values:
        values = sorted(
            {
                str(record["route_plan_sha256"])
                for record in stats.get("call_records", [])
                if record.get("route_plan_sha256")
            }
        )
    if not values:
        return None, []
    digest = hashlib.sha256("\n".join(values).encode()).hexdigest()
    return digest, values


def _cases(suite: dict) -> list[dict]:
    if suite.get("cases"):
        return [dict(case) for case in suite["cases"]]
    return [
        {
            "prompt_id": suite.get("prompt_id", "smoke"),
            "prompt": suite["prompt"],
            "seed": int(suite["seed"]),
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/inferhub/rag_method_21.yaml")
    parser.add_argument("--suite", default="configs/rag_smoke_methods.json")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    suite = json.loads((ROOT / args.suite).read_text(encoding="utf-8"))
    methods = suite["methods"][args.shard_index :: args.shard_count]
    if not methods:
        raise ValueError("empty method shard")
    cases = _cases(suite)
    latent_frames = int(suite.get("latent_frames", 21))
    base = yaml.safe_load((ROOT / args.base_config).read_text(encoding="utf-8"))
    base_output = Path(os.environ["INFER_OUTPUT_DIR"]) / "base_load"
    base_output.mkdir(parents=True, exist_ok=True)
    base["output_folder"] = str(base_output)
    empty_prompts = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    empty_prompts.close()
    base["data_path"] = empty_prompts.name
    base["inference_iter"] = 0
    base["num_output_frames"] = 21
    base["sparse_history"]["method"] = "rag_dense"
    base["sparse_history"]["history_density"] = 1.0
    temporary = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(base, temporary, sort_keys=False)
    temporary.close()

    from scripts.run_longlive_sparse import run_config

    namespace = run_config(temporary.name)
    pipeline = namespace.get("pipeline")
    if pipeline is None:
        raise RuntimeError("loaded pipeline was not exposed by run_longlive_sparse")

    from adapters.longlive_sparse.config import SparseHistoryConfig
    from adapters.longlive_sparse.methods import method_spec
    from adapters.longlive_sparse.stats import SparseRunStats
    from utils.misc import set_seed

    output_root = Path(os.environ["INFER_OUTPUT_DIR"])
    case_states = []
    for method in methods:
        backend = suite.get("method_backends", {}).get(method, suite["backend"])
        config = SparseHistoryConfig(
            method=method,
            backend=backend,
            history_density=float(suite["history_density"]),
            refresh_policy=str(suite.get("refresh_policy", "per_chunk")),
            rope_policy=str(suite.get("rope_policy", "upstream_zero")),
            fail_on_fallback=True,
            record_per_call=bool(suite.get("record_per_call", latent_frames <= 39)),
            method_params=dict(suite.get("method_params", {}).get(method, {})),
        )
        pipeline.sparse_history_config = config
        pipeline.sparse_history_archive.config = config
        for module in pipeline.sparse_history_modules:
            module.sparse_config = config
            module.clear_selection_cache()
        for case in cases:
            seed = int(case["seed"])
            prompt_id = str(case["prompt_id"])
            case_id = f"{method}__{prompt_id}__s{seed}"
            case_dir = output_root / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            pipeline.sparse_history_aggregate_stats = SparseRunStats(method=method)
            pipeline.sparse_history_completed_runs = []
            set_seed(seed)
            noise = torch.randn(
                [1, latent_frames, 16, 60, 104],
                device=next(pipeline.generator.parameters()).device,
                dtype=torch.bfloat16,
            )
            try:
                video, latents = pipeline.inference(
                    noise=noise,
                    text_prompts=[case["prompt"]],
                    return_latents=True,
                    low_memory=True,
                    profile=True,
                )
                if not torch.isfinite(latents).all() or not torch.isfinite(video).all():
                    raise FloatingPointError("video or latents contain NaN/Inf")
                frames = (
                    255 * rearrange(video, "b t c h w -> b t h w c").cpu()
                ).clamp(0, 255).to(torch.uint8)
                video_path = case_dir / "video.mp4"
                write_video(str(video_path), frames[0], fps=16)
                torch.save(latents.detach().cpu(), case_dir / "latents.pt")
                stats = pipeline.sparse_history_aggregate_stats.as_dict()
                route_digest, route_shas = _route_digest(stats)
                (case_dir / "sparse_history_stats.json").write_text(
                    json.dumps(stats, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                case_config = {
                    "method": method,
                    "routing_stage": method_spec(method).routing_stage,
                    "backend": backend,
                    "history_density": suite["history_density"],
                    "latent_frames": latent_frames,
                    "prompt_id": prompt_id,
                    "prompt": case["prompt"],
                    "seed": seed,
                    "method_params": config.method_params,
                }
                (case_dir / "case_config.json").write_text(
                    json.dumps(case_config, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                state = {
                    "id": case_id,
                    "method": method,
                    "status": "pass",
                    "routing_stage": method_spec(method).routing_stage,
                    "backend": stats.get("attention_backend"),
                    "route_plan_sha256": route_digest,
                    "route_plan_sha256s": route_shas,
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "video": str(video_path),
                    "video_sha256": _sha256(video_path),
                    "pixel_frames": int(frames.shape[1]),
                    "stats": str(case_dir / "sparse_history_stats.json"),
                    "config": str(case_dir / "case_config.json"),
                    "failed_calls": stats.get("failed_calls", 0),
                    "fallback_calls": stats.get("dense_fallback_calls", 0),
                    "nan_calls": 0,
                }
                if state["failed_calls"] or state["fallback_calls"]:
                    raise RuntimeError("successful generation reported failed/fallback calls")
                case_states.append(state)
            except Exception as error:
                state = {
                    "id": case_id,
                    "method": method,
                    "status": "fail",
                    "routing_stage": method_spec(method).routing_stage,
                    "backend": backend,
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "failure_reason": f"{type(error).__name__}: {error}",
                }
                (case_dir / "failure.json").write_text(
                    json.dumps(state, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                case_states.append(state)
            finally:
                pipeline.vae.model.clear_cache()
                torch.cuda.empty_cache()
                gc.collect()
    (output_root / f"shard_{args.shard_index}_states.json").write_text(
        json.dumps({"cases": case_states}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if any(case["status"] == "fail" for case in case_states):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
