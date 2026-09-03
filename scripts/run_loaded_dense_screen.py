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

from adapters.longlive_sparse.case_identity import (
    build_case_identity,
    resolve_experiment_provenance,
)
from adapters.longlive_sparse.video_decode import (
    decode_latents_chunked_exact,
    expected_pixel_frames,
)


RUNTIME_SPECS = {
    "native_dense": {
        "backend": "packed_fa2",
        "history_density": 1.0,
        "refresh_policy": "not_applicable",
        "rope_policy": "not_applicable",
        "routing_stage": "N/A",
    },
    "native_block": {
        "backend": "grouped_fa2",
        "history_density": 0.25,
        "refresh_policy": "per_chunk",
        "rope_policy": "upstream_zero",
        "routing_stage": "N/A",
    },
    "rag_dense": {
        "backend": "packed_fa2",
        "history_density": 1.0,
        "refresh_policy": "per_chunk",
        "rope_policy": "upstream_zero",
        "routing_stage": "post-transfer",
    },
}


class DenseAttentionProfiler:
    """Accumulate native self-attention time with the sparse runner's timing rule."""

    def __init__(self) -> None:
        self.total_s = 0.0
        self.calls = 0
        self._installed = False

    def reset(self) -> None:
        self.total_s = 0.0
        self.calls = 0

    def install(self) -> None:
        if self._installed:
            return
        modules = []
        from wan.modules import causal_model

        modules.append(causal_model)
        try:
            from wan.modules import causal_model_infinity

            modules.append(causal_model_infinity)
        except ImportError:
            pass
        for module in modules:
            original = module.attention

            def timed_attention(*args, _original=original, **kwargs):
                started = time.perf_counter()
                output = _original(*args, **kwargs)
                query = args[0] if args else kwargs.get("q")
                if isinstance(query, torch.Tensor) and query.is_cuda:
                    torch.cuda.synchronize(query.device)
                self.total_s += time.perf_counter() - started
                self.calls += 1
                return output

            module.attention = timed_attention
        self._installed = True


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
    parser.add_argument("--experiment-commit")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")

    output_root = Path(os.environ["INFER_OUTPUT_DIR"])
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / args.candidates).read_text(encoding="utf-8"))
    commit, execution_commit, execution_change_scope = resolve_experiment_provenance(
        args.experiment_commit or manifest.get("experiment_commit"), repo_root=ROOT
    )
    seeds = [int(value) for value in manifest.get("seeds", [])]
    if manifest.get("cases"):
        all_candidates = [
            {**case, "_seeds": [int(case["seed"])]} for case in manifest["cases"]
        ]
    else:
        all_candidates = [
            {**candidate, "_seeds": seeds} for candidate in manifest["candidates"]
        ]
    candidates = all_candidates[args.shard_index :: args.shard_count]
    if not candidates:
        raise ValueError("empty Dense screen shard")
    task_count = sum(len(candidate["_seeds"]) for candidate in candidates)
    load_started = time.perf_counter()
    pipeline = _load_pipeline(args.runtime, ROOT / args.base_config, output_root)
    model_load_s = time.perf_counter() - load_started
    load_amortized_s = model_load_s / task_count
    runtime_spec = RUNTIME_SPECS[args.runtime]
    dense_attention_profiler = DenseAttentionProfiler()
    if args.runtime == "native_dense":
        dense_attention_profiler.install()

    from utils.misc import set_seed
    from wan.modules.attention import attention_backend

    case_states = []
    for candidate in candidates:
        for seed in candidate["_seeds"]:
            latent_frames = int(candidate.get("latent_frames", args.latent_frames))
            identity = build_case_identity(
                commit=commit,
                method=args.runtime,
                prompt_id=candidate["prompt_id"],
                prompt=candidate["prompt"],
                seed=seed,
                latent_frames=latent_frames,
                history_density=runtime_spec["history_density"],
                rope_policy=runtime_spec["rope_policy"],
                refresh_policy=runtime_spec["refresh_policy"],
                backend=runtime_spec["backend"],
            )
            case_id = identity["id"]
            case_dir = output_root / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            state_path = case_dir / "case_metrics.json"
            if state_path.is_file():
                existing = json.loads(state_path.read_text(encoding="utf-8"))
                video_path = Path(str(existing.get("video", "")))
                latent_path = case_dir / "latents.pt"
                latent = None
                if latent_path.is_file():
                    latent = torch.load(latent_path, map_location="cpu", weights_only=True)
                if (
                    existing.get("status") == "pass"
                    and existing.get("case_key_sha256")
                    == identity["case_key_sha256"]
                    and video_path.is_file()
                    and latent is not None
                    and tuple(latent.shape)
                    == (1, latent_frames, 16, 60, 104)
                    and bool(torch.isfinite(latent).all())
                    and _sha256(video_path) == existing.get("video_sha256")
                    and _decoded_frames(video_path)
                    == existing.get("decoded_frames", existing.get("pixel_frames"))
                ):
                    existing["resume_action"] = "reused_verified_success"
                    case_states.append(existing)
                    continue
            started = time.perf_counter()
            try:
                dense_attention_profiler.reset()
                if hasattr(pipeline, "sparse_history_config"):
                    from adapters.longlive_sparse.stats import SparseRunStats

                    pipeline.sparse_history_aggregate_stats = SparseRunStats(
                        method=pipeline.sparse_history_config.method
                    )
                    pipeline.sparse_history_completed_runs = []
                set_seed(seed)
                torch.cuda.reset_peak_memory_stats()
                noise = torch.randn(
                    [1, latent_frames, 16, 60, 104],
                    device=next(pipeline.generator.parameters()).device,
                    dtype=torch.bfloat16,
                )
                inference_kwargs = {
                    "noise": noise,
                    "text_prompts": [candidate["prompt"]],
                    "return_latents": True,
                    "low_memory": True,
                    "profile": True,
                }
                if args.runtime == "rag_dense" and latent_frames > 120:
                    inference_kwargs["skip_vae_decode"] = True
                video, latents = pipeline.inference(**inference_kwargs)
                decode_mode = "upstream"
                if video is None:
                    video = decode_latents_chunked_exact(
                        pipeline.vae,
                        latents,
                        chunk_size=120,
                    )
                    video = (video * 0.5 + 0.5).clamp(0, 1)
                    decode_mode = "cache_continuous_chunked_120"
                if not torch.isfinite(video).all() or not torch.isfinite(latents).all():
                    raise FloatingPointError("video or latents contain NaN/Inf")
                torch.save(latents.detach().cpu(), case_dir / "latents.pt")
                frames = (
                    255
                    * rearrange(video, "b t c h w -> b t h w c").cpu()
                ).clamp(0, 255).to(torch.uint8)[0]
                video_path = case_dir / "video.mp4"
                write_video(str(video_path), frames, fps=16)
                decoded_frames = _decoded_frames(video_path)
                expected_frames = expected_pixel_frames(latent_frames)
                if decoded_frames != expected_frames:
                    raise RuntimeError(
                        f"decoded frame count {decoded_frames} != expected {expected_frames}"
                    )
                _save_review_frames(frames, case_dir)
                stats = (
                    pipeline.sparse_history_aggregate_stats.as_dict()
                    if hasattr(pipeline, "sparse_history_aggregate_stats")
                    else {
                        "method": "native_dense",
                        "routing_stage": "N/A",
                        "attention_backend": attention_backend(),
                        "history_transfer_density": None,
                        "history_density": 1.0,
                        "history_pair_density": 1.0,
                        "global_executed_density": 1.0,
                        "failed_calls": 0,
                        "dense_fallback_calls": 0,
                        "timing": {
                            "attention_s": dense_attention_profiler.total_s,
                            "routing_s": 0.0,
                            "cpu_gather_s": 0.0,
                            "h2d_s": 0.0,
                            "rope_s": 0.0,
                        },
                        "native_attention_calls": dense_attention_profiler.calls,
                    }
                )
                route_digest, route_shas = _route_digest(stats)
                stats_path = case_dir / "sparse_history_stats.json"
                stats_path.write_text(
                    json.dumps(stats, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                config_payload = {
                    **identity,
                    "runtime": args.runtime,
                    "method": args.runtime,
                    "routing_stage": runtime_spec["routing_stage"],
                    "backend": runtime_spec["backend"],
                    "history_density": runtime_spec["history_density"],
                    "refresh_policy": runtime_spec["refresh_policy"],
                    "rope_policy": runtime_spec["rope_policy"],
                    "prompt_id": candidate["prompt_id"],
                    "prompt": candidate["prompt"],
                    "seed": seed,
                    "latent_frames": latent_frames,
                    "execution_commit": execution_commit,
                    "execution_change_scope": execution_change_scope,
                    "decode_mode": decode_mode,
                }
                config_path = case_dir / "case_config.json"
                config_path.write_text(
                    json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                elapsed_s = time.perf_counter() - started
                metrics = {
                    **identity,
                    "status": "pass",
                    "method": args.runtime,
                    "runtime": args.runtime,
                    "routing_stage": runtime_spec["routing_stage"],
                    "prompt_id": candidate["prompt_id"],
                    "category": candidate["category"],
                    "prompt": candidate["prompt"],
                    "seed": seed,
                    "latent_frames": latent_frames,
                    "pixel_frames": int(frames.shape[0]),
                    "decoded_frames": decoded_frames,
                    "attention_backend": attention_backend(),
                    "backend": runtime_spec["backend"],
                    "observed_attention_backend": stats.get(
                        "attention_backend", attention_backend()
                    ),
                    "history_density": runtime_spec["history_density"],
                    "refresh_policy": runtime_spec["refresh_policy"],
                    "rope_policy": runtime_spec["rope_policy"],
                    "route_plan_sha256": route_digest,
                    "route_plan_sha256s": route_shas,
                    "elapsed_s": elapsed_s,
                    "end_to_end_s": elapsed_s,
                    "model_load_s_total": model_load_s,
                    "model_load_s_amortized": load_amortized_s,
                    "end_to_end_with_amortized_load_s": elapsed_s + load_amortized_s,
                    "peak_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
                    "video": str(video_path),
                    "video_sha256": _sha256(video_path),
                    "history_pair_density": stats.get("history_pair_density", 1.0),
                    "history_transfer_density": stats.get("history_transfer_density"),
                    "global_executed_density": stats.get("global_executed_density", 1.0),
                    "candidate_transfer_bytes": stats.get("candidate_transfer_bytes", 0),
                    "transferred_bytes": stats.get("transferred_bytes", 0),
                    "archive_bytes": stats.get("archive_bytes", 0),
                    "index_bytes": stats.get("index_bytes", 0),
                    "index_transfer_bytes": stats.get("index_transfer_bytes", 0),
                    "staging_padding_tokens": stats.get("staging_padding_tokens", 0),
                    "selected_history_tokens": stats.get("selected_history_tokens", 0),
                    "candidate_history_tokens": stats.get("candidate_history_tokens", 0),
                    "executed_qk_pairs": stats.get("executed_qk_pairs"),
                    "dense_qk_pairs": stats.get("dense_qk_pairs"),
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
                    "execution_commit": execution_commit,
                    "execution_change_scope": execution_change_scope,
                    "decode_mode": decode_mode,
                }
                state_path.write_text(
                    json.dumps(metrics, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                case_states.append(metrics)
            except Exception as error:
                state = {
                    **identity,
                    "status": "fail",
                    "method": args.runtime,
                    "runtime": args.runtime,
                    "routing_stage": runtime_spec["routing_stage"],
                    "backend": runtime_spec["backend"],
                    "history_density": runtime_spec["history_density"],
                    "refresh_policy": runtime_spec["refresh_policy"],
                    "rope_policy": runtime_spec["rope_policy"],
                    "prompt_id": candidate["prompt_id"],
                    "category": candidate["category"],
                    "seed": seed,
                    "failure_reason": f"{type(error).__name__}: {error}",
                    "elapsed_s": time.perf_counter() - started,
                    "execution_commit": execution_commit,
                    "execution_change_scope": execution_change_scope,
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
        "commit": commit,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "model_load_s": model_load_s,
        "formal_prompts_frozen": manifest.get("status") == "frozen_basic_477",
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
