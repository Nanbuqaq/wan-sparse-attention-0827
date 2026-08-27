#!/usr/bin/env python3
"""Load LongLive-RAG once, then run a shard of distinct sparse methods."""

from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
from pathlib import Path

import torch
import yaml
from einops import rearrange
from torchvision.io import write_video


ROOT = Path(__file__).resolve().parents[1]


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
    base = yaml.safe_load((ROOT / args.base_config).read_text(encoding="utf-8"))
    base_output = Path(os.environ["INFER_OUTPUT_DIR"]) / "base_load"
    base_output.mkdir(parents=True, exist_ok=True)
    base["output_folder"] = str(base_output)
    base["data_path"] = "configs/prompts/smoke.txt"
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
    from adapters.longlive_sparse.stats import SparseRunStats
    from utils.misc import set_seed

    output_root = Path(os.environ["INFER_OUTPUT_DIR"])
    case_states = []
    for method in methods:
        case_dir = output_root / method
        case_dir.mkdir(parents=True, exist_ok=True)
        config = SparseHistoryConfig(
            method=method,
            backend=suite["backend"],
            history_density=float(suite["history_density"]),
            refresh_policy="per_chunk",
            rope_policy="upstream_zero",
        )
        pipeline.sparse_history_config = config
        pipeline.sparse_history_archive.config = config
        pipeline.sparse_history_aggregate_stats = SparseRunStats(method=method)
        pipeline.sparse_history_completed_runs = []
        for module in pipeline.sparse_history_modules:
            module.sparse_config = config
            module.clear_selection_cache()
        set_seed(int(suite["seed"]))
        noise = torch.randn(
            [1, int(suite["latent_frames"]), 16, 60, 104],
            device=next(pipeline.generator.parameters()).device,
            dtype=torch.bfloat16,
        )
        try:
            video, latents = pipeline.inference(
                noise=noise,
                text_prompts=[suite["prompt"]],
                return_latents=True,
                low_memory=True,
                profile=True,
            )
            if not torch.isfinite(latents).all() or not torch.isfinite(video).all():
                raise FloatingPointError("video or latents contain NaN/Inf")
            frames = (255 * rearrange(video, "b t c h w -> b t h w c").cpu()).clamp(0, 255).to(torch.uint8)
            video_path = case_dir / "rank0-0-0_lora.mp4"
            write_video(str(video_path), frames[0], fps=16)
            torch.save(latents.cpu(), case_dir / "latents.pt")
            stats = pipeline.sparse_history_aggregate_stats.as_dict()
            (case_dir / "sparse_history_stats.json").write_text(
                json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            case_states.append({"id": method, "status": "pass", "video": str(video_path), "stats": str(case_dir / "sparse_history_stats.json")})
        except Exception as error:
            case_states.append({"id": method, "status": "fail", "failure_reason": f"{type(error).__name__}: {error}"})
        finally:
            pipeline.vae.model.clear_cache()
            torch.cuda.empty_cache()
            gc.collect()
    (output_root / f"shard_{args.shard_index}_states.json").write_text(
        json.dumps({"cases": case_states}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if any(case["status"] == "fail" for case in case_states):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
