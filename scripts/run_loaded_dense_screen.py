#!/usr/bin/env python3
"""Load one Dense runtime once and generate all two-seed prompt-screen cases."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import av
import torch
import yaml
from einops import rearrange
from PIL import Image
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


def _decoded_frames(path: Path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def _route_digest(stats: dict) -> tuple[str | None, list[str]]:
    values = sorted(stats.get("route_plan_sha256_counts", {}))
    if not values:
        return None, []
    return hashlib.sha256("\n".join(values).encode()).hexdigest(), values


def _save_review_frames(frames: torch.Tensor, case_dir: Path) -> None:
    for name, index in (
        ("first", 0),
        ("middle", frames.shape[0] // 2),
        ("last", frames.shape[0] - 1),
    ):
        Image.fromarray(frames[index].numpy()).save(case_dir / f"review_{name}.png")


def _load_pipeline(runtime: str, base_config_path: Path, output_root: Path):
    config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    empty_prompts = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    empty_prompts.close()
    config["data_path"] = empty_prompts.name
    config["output_folder"] = str(output_root / "base_load")
    config["inference_iter"] = 0
    config["num_output_frames"] = 21
    if runtime == "rag_dense":
        config["runtime_mode"] = "rag_sparse"
        config["sparse_history"] = {
            "method": "rag_dense",
            "backend": "packed_fa2",
            "history_density": 1.0,
            "refresh_policy": "per_chunk",
            "rope_policy": "upstream_zero",
            "fail_on_fallback": True,
            "record_per_call": False,
        }
    elif runtime == "native_block":
        config["runtime_mode"] = "rag_sparse"
        config["model_kwargs"]["memory_size"] = 0
        config["model_kwargs"]["recent_exclude"] = 0
        config["sparse_history"] = {
            "method": "native_block",
            "backend": "grouped_fa2",
            "history_density": 0.25,
            "recent_exact_frames": 3,
            "refresh_policy": "per_chunk",
            "rope_policy": "upstream_zero",
            "fail_on_fallback": True,
            "record_per_call": False,
        }
    elif runtime == "native_dense":
        config["runtime_mode"] = "native_dense"
        config.pop("sparse_history", None)
    else:
        raise ValueError(f"unsupported Dense screen runtime: {runtime}")
    temporary = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(config, temporary, sort_keys=False)
    temporary.close()
    from scripts.run_longlive_sparse import run_config

    namespace = run_config(temporary.name)
    pipeline = namespace.get("pipeline")
    if pipeline is None:
        raise RuntimeError("loaded pipeline was not exposed")
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime",
        choices=("native_dense", "native_block", "rag_dense"),
        required=True,
    )
    parser.add_argument("--base-config", required=True)
    parser.add_argument(
        "--candidates", default="configs/prompts/dense_candidates.json"
    )
    parser.add_argument("--latent-frames", type=int, default=120)
    args = parser.parse_args()

    output_root = Path(os.environ["INFER_OUTPUT_DIR"])
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / args.candidates).read_text(encoding="utf-8"))
    seeds = [int(value) for value in manifest["seeds"]]
    pipeline = _load_pipeline(args.runtime, ROOT / args.base_config, output_root)

    from utils.misc import set_seed
    from wan.modules.attention import attention_backend

    case_states = []
    for candidate in manifest["candidates"]:
        for seed in seeds:
            case_id = f"{args.runtime}__{candidate['prompt_id']}__s{seed}"
            case_dir = output_root / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            state_path = case_dir / "case_metrics.json"
            if state_path.is_file():
                existing = json.loads(state_path.read_text(encoding="utf-8"))
                video_path = Path(str(existing.get("video", "")))
                latent_path = case_dir / "latents.pt"
                if (
                    existing.get("status") == "pass"
                    and video_path.is_file()
                    and latent_path.is_file()
                    and _sha256(video_path) == existing.get("video_sha256")
                    and _decoded_frames(video_path)
                    == existing.get("decoded_frames", existing.get("pixel_frames"))
                ):
                    existing["resume_action"] = "reused_verified_success"
                    case_states.append(existing)
                    continue
            started = time.perf_counter()
            try:
                if hasattr(pipeline, "sparse_history_config"):
                    from adapters.longlive_sparse.stats import SparseRunStats

                    pipeline.sparse_history_aggregate_stats = SparseRunStats(
                        method=pipeline.sparse_history_config.method
                    )
                    pipeline.sparse_history_completed_runs = []
                set_seed(seed)
                torch.cuda.reset_peak_memory_stats()
                noise = torch.randn(
                    [1, args.latent_frames, 16, 60, 104],
                    device=next(pipeline.generator.parameters()).device,
                    dtype=torch.bfloat16,
                )
                video, latents = pipeline.inference(
                    noise=noise,
                    text_prompts=[candidate["prompt"]],
                    return_latents=True,
                    low_memory=True,
                    profile=True,
                )
                if not torch.isfinite(video).all() or not torch.isfinite(latents).all():
                    raise FloatingPointError("video or latents contain NaN/Inf")
                frames = (
                    255
                    * rearrange(video, "b t c h w -> b t h w c").cpu()
                ).clamp(0, 255).to(torch.uint8)[0]
                video_path = case_dir / "video.mp4"
                write_video(str(video_path), frames, fps=16)
                decoded_frames = _decoded_frames(video_path)
                expected_frames = 4 * args.latent_frames - 3
                if decoded_frames != expected_frames:
                    raise RuntimeError(
                        f"decoded frame count {decoded_frames} != expected {expected_frames}"
                    )
                torch.save(latents.detach().cpu(), case_dir / "latents.pt")
                _save_review_frames(frames, case_dir)
                stats = (
                    pipeline.sparse_history_aggregate_stats.as_dict()
                    if hasattr(pipeline, "sparse_history_aggregate_stats")
                    else {
                        "method": "native_dense",
                        "routing_stage": "N/A",
                        "history_transfer_density": None,
                        "history_pair_density": 1.0,
                        "global_executed_density": 1.0,
                        "failed_calls": 0,
                        "dense_fallback_calls": 0,
                    }
                )
                route_digest, route_shas = _route_digest(stats)
                stats_path = case_dir / "sparse_history_stats.json"
                stats_path.write_text(
                    json.dumps(stats, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                config_payload = {
                    "runtime": args.runtime,
                    "routing_stage": (
                        "N/A" if args.runtime != "rag_dense" else "post-transfer"
                    ),
                    "prompt_id": candidate["prompt_id"],
                    "prompt": candidate["prompt"],
                    "seed": seed,
                    "latent_frames": args.latent_frames,
                }
                config_path = case_dir / "case_config.json"
                config_path.write_text(
                    json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                metrics = {
                    "case_id": case_id,
                    "status": "pass",
                    "runtime": args.runtime,
                    "routing_stage": (
                        "post-transfer" if args.runtime == "rag_dense" else "N/A"
                    ),
                    "prompt_id": candidate["prompt_id"],
                    "category": candidate["category"],
                    "prompt": candidate["prompt"],
                    "seed": seed,
                    "latent_frames": args.latent_frames,
                    "pixel_frames": int(frames.shape[0]),
                    "decoded_frames": decoded_frames,
                    "attention_backend": attention_backend(),
                    "backend": stats.get("attention_backend", attention_backend()),
                    "route_plan_sha256": route_digest,
                    "route_plan_sha256s": route_shas,
                    "elapsed_s": time.perf_counter() - started,
                    "peak_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
                    "video": str(video_path),
                    "video_sha256": _sha256(video_path),
                    "history_pair_density": stats.get("history_pair_density", 1.0),
                    "history_transfer_density": stats.get("history_transfer_density"),
                    "global_executed_density": stats.get("global_executed_density", 1.0),
                    "candidate_transfer_bytes": stats.get("candidate_transfer_bytes", 0),
                    "transferred_bytes": stats.get("transferred_bytes", 0),
                    "attention_s": stats.get("timing", {}).get("attention_s"),
                    "routing_s": stats.get("timing", {}).get("routing_s"),
                    "cpu_gather_s": stats.get("timing", {}).get("cpu_gather_s"),
                    "h2d_s": stats.get("timing", {}).get("h2d_s"),
                    "rope_s": stats.get("timing", {}).get("rope_s"),
                    "failed_calls": stats.get("failed_calls", 0),
                    "fallback_calls": stats.get("dense_fallback_calls", 0),
                    "nan_calls": 0,
                    "stats": str(stats_path),
                    "stats_summary": stats,
                    "config": str(config_path),
                }
                state_path.write_text(
                    json.dumps(metrics, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                case_states.append(metrics)
            except Exception as error:
                state = {
                    "case_id": case_id,
                    "status": "fail",
                    "runtime": args.runtime,
                    "prompt_id": candidate["prompt_id"],
                    "category": candidate["category"],
                    "seed": seed,
                    "failure_reason": f"{type(error).__name__}: {error}",
                    "elapsed_s": time.perf_counter() - started,
                }
                state_path.write_text(
                    json.dumps(state, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                case_states.append(state)
            finally:
                pipeline.vae.model.clear_cache()
                torch.cuda.empty_cache()
                gc.collect()

    payload = {
        "runtime": args.runtime,
        "formal_prompts_frozen": False,
        "sparse_results_used": False,
        "cases": case_states,
    }
    (output_root / "dense_screen_states.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"runtime": args.runtime, "cases": len(case_states)}, indent=2))
    if any(case["status"] == "fail" for case in case_states):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
