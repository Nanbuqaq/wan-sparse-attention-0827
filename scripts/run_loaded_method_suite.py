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
import time
from pathlib import Path

import av
import torch
import yaml
from einops import rearrange
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
    parser.add_argument("--method-params-file")
    parser.add_argument("--experiment-commit")
    parser.add_argument("--shard-axis", choices=("method", "case"), default="method")
    args = parser.parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    suite = json.loads((ROOT / args.suite).read_text(encoding="utf-8"))
    commit, execution_commit, execution_change_scope = resolve_experiment_provenance(
        args.experiment_commit or suite.get("experiment_commit"), repo_root=ROOT
    )
    all_methods = list(suite["methods"])
    cases = _cases(suite)
    if args.shard_axis == "method":
        methods = all_methods[args.shard_index :: args.shard_count]
        task_count = len(methods) * len(cases)
    else:
        methods = all_methods
        task_count = sum(
            index % args.shard_count == args.shard_index
            for index in range(len(all_methods) * len(cases))
        )
    if not methods or not task_count:
        raise ValueError("empty method/case shard")
    if args.method_params_file:
        frozen_params = json.loads(
            Path(args.method_params_file).read_text(encoding="utf-8")
        )
        if frozen_params.get("status") not in {
            "frozen_before_method_smoke",
            "frozen_before_formal_long_video",
        }:
            raise ValueError("method parameter file is not frozen for method smoke")
        suite.setdefault("method_params", {}).update(
            frozen_params.get("method_params", {})
        )
    default_latent_frames = int(suite.get("latent_frames", 21))
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

    load_started = time.perf_counter()
    from scripts.run_longlive_sparse import run_config

    namespace = run_config(temporary.name)
    pipeline = namespace.get("pipeline")
    if pipeline is None:
        raise RuntimeError("loaded pipeline was not exposed by run_longlive_sparse")
    model_load_s = time.perf_counter() - load_started
    load_amortized_s = model_load_s / task_count

    from adapters.longlive_sparse.config import SparseHistoryConfig
    from adapters.longlive_sparse.methods import method_spec
    from adapters.longlive_sparse.stats import SparseRunStats
    from utils.misc import set_seed

    output_root = Path(os.environ["INFER_OUTPUT_DIR"])
    case_states = []
    for method in methods:
        method_index = all_methods.index(method)
        for case_index, case in enumerate(cases):
            task_index = method_index * len(cases) + case_index
            if (
                args.shard_axis == "case"
                and task_index % args.shard_count != args.shard_index
            ):
                continue
            latent_frames = int(case.get("latent_frames", default_latent_frames))
            backend = str(
                case.get(
                    "backend",
                    suite.get("method_backends", {}).get(method, suite["backend"]),
                )
            )
            history_density = float(
                case.get("history_density", suite["history_density"])
            )
            refresh_policy = str(
                case.get("refresh_policy", suite.get("refresh_policy", "per_chunk"))
            )
            rope_policy = str(
                case.get("rope_policy", suite.get("rope_policy", "upstream_zero"))
            )
            method_params = dict(suite.get("method_params", {}).get(method, {}))
            method_params.update(case.get("method_params", {}))
            config = SparseHistoryConfig(
                method=method,
                backend=backend,
                history_density=history_density,
                refresh_policy=refresh_policy,
                rope_policy=rope_policy,
                fail_on_fallback=True,
                record_per_call=bool(
                    case.get(
                        "record_per_call",
                        suite.get("record_per_call", latent_frames <= 39),
                    )
                ),
                method_params=method_params,
            )
            pipeline.sparse_history_config = config
            pipeline.sparse_history_archive.config = config
            for module in pipeline.sparse_history_modules:
                module.sparse_config = config
                module.clear_selection_cache()
            seed = int(case["seed"])
            prompt_id = str(case["prompt_id"])
            identity = build_case_identity(
                commit=commit,
                method=method,
                prompt_id=prompt_id,
                prompt=case["prompt"],
                seed=seed,
                latent_frames=latent_frames,
                history_density=history_density,
                rope_policy=config.rope_policy,
                refresh_policy=config.refresh_policy,
                backend=backend,
            )
            case_id = identity["id"]
            case_dir = output_root / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            state_path = case_dir / "case_state.json"
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
                    and tuple(latent.shape) == (1, latent_frames, 16, 60, 104)
                    and bool(torch.isfinite(latent).all())
                    and _sha256(video_path) == existing.get("video_sha256")
                    and _decoded_frames(video_path) == existing.get("decoded_frames")
                ):
                    existing["resume_action"] = "reused_verified_success"
                    case_states.append(existing)
                    continue
            elif latent_frames <= 39:
                video_path = case_dir / "video.mp4"
                latent_path = case_dir / "latents.pt"
                stats_path = case_dir / "sparse_history_stats.json"
                config_path = case_dir / "case_config.json"
                if all(
                    path.is_file()
                    for path in (video_path, latent_path, stats_path, config_path)
                ):
                    stats = json.loads(stats_path.read_text(encoding="utf-8"))
                    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
                    latent = torch.load(latent_path, map_location="cpu", weights_only=True)
                    decoded_frames = -1
                    if not stats.get("failed_calls", 0) and not stats.get(
                        "dense_fallback_calls", 0
                    ):
                        decoded_frames = _decoded_frames(video_path)
                        if decoded_frames != 4 * latent_frames - 3:
                            decoded_frames = -1
                    if (
                        not stats.get("failed_calls", 0)
                        and not stats.get("dense_fallback_calls", 0)
                        and decoded_frames == 4 * latent_frames - 3
                        and saved_config.get("case_key_sha256")
                        == identity["case_key_sha256"]
                        and tuple(latent.shape) == (1, latent_frames, 16, 60, 104)
                        and bool(torch.isfinite(latent).all())
                    ):
                        route_digest, route_shas = _route_digest(stats)
                        recovered = {
                            **identity,
                            "method": method,
                            "status": "pass",
                            "routing_stage": method_spec(method).routing_stage,
                            "backend": backend,
                            "observed_attention_backend": stats.get(
                                "attention_backend"
                            ),
                            "history_density": history_density,
                            "refresh_policy": config.refresh_policy,
                            "rope_policy": config.rope_policy,
                            "latent_frames": latent_frames,
                            "route_plan_sha256": route_digest,
                            "route_plan_sha256s": route_shas,
                            "prompt_id": prompt_id,
                            "seed": seed,
                            "video": str(video_path),
                            "video_sha256": _sha256(video_path),
                            "decoded_frames": decoded_frames,
                            "stats": str(stats_path),
                            "config": str(config_path),
                            "failed_calls": 0,
                            "fallback_calls": 0,
                            "nan_calls": 0,
                            "history_pair_density": stats.get("history_pair_density"),
                            "history_transfer_density": stats.get("history_transfer_density"),
                            "global_executed_density": stats.get("global_executed_density"),
                            "candidate_transfer_bytes": stats.get("candidate_transfer_bytes"),
                            "transferred_bytes": stats.get("transferred_bytes"),
                            "archive_bytes": stats.get("archive_bytes"),
                            "index_bytes": stats.get("index_bytes"),
                            "index_transfer_bytes": stats.get("index_transfer_bytes"),
                            "query_summary_bytes": stats.get("query_summary_bytes"),
                            "staging_padding_tokens": stats.get("staging_padding_tokens"),
                            "selected_history_tokens": stats.get("selected_history_tokens"),
                            "candidate_history_tokens": stats.get("candidate_history_tokens"),
                            "executed_qk_pairs": stats.get("executed_qk_pairs"),
                            "dense_qk_pairs": stats.get("dense_qk_pairs"),
                            "model_load_s_total": model_load_s,
                            "model_load_s_amortized": load_amortized_s,
                            "attention_s": stats.get("timing", {}).get("attention_s"),
                            "routing_s": stats.get("timing", {}).get("routing_s"),
                            "q_summary_s": stats.get("timing", {}).get("q_summary_s"),
                            "d2h_s": stats.get("timing", {}).get("d2h_s"),
                            "cpu_gather_s": stats.get("timing", {}).get("cpu_gather_s"),
                            "h2d_s": stats.get("timing", {}).get("h2d_s"),
                            "rope_s": stats.get("timing", {}).get("rope_s"),
                            "resume_action": "recovered_verified_artifacts",
                        }
                        state_path.write_text(
                            json.dumps(recovered, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                        case_states.append(recovered)
                        continue
            pipeline.sparse_history_aggregate_stats = SparseRunStats(method=method)
            pipeline.sparse_history_completed_runs = []
            set_seed(seed)
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            noise = torch.randn(
                [1, latent_frames, 16, 60, 104],
                device=next(pipeline.generator.parameters()).device,
                dtype=torch.bfloat16,
            )
            try:
                defer_vae_decode = latent_frames > 120
                video, latents = pipeline.inference(
                    noise=noise,
                    text_prompts=[case["prompt"]],
                    return_latents=True,
                    low_memory=True,
                    profile=True,
                    skip_vae_decode=defer_vae_decode,
                )
                decode_mode = "upstream"
                if video is None:
                    video = decode_latents_chunked_exact(
                        pipeline.vae,
                        latents,
                        chunk_size=120,
                    )
                    video = (video * 0.5 + 0.5).clamp(0, 1)
                    decode_mode = "cache_continuous_chunked_120"
                if not torch.isfinite(latents).all() or not torch.isfinite(video).all():
                    raise FloatingPointError("video or latents contain NaN/Inf")
                torch.save(latents.detach().cpu(), case_dir / "latents.pt")
                frames = (
                    255 * rearrange(video, "b t c h w -> b t h w c").cpu()
                ).clamp(0, 255).to(torch.uint8)
                video_path = case_dir / "video.mp4"
                write_video(str(video_path), frames[0], fps=16)
                decoded_frames = _decoded_frames(video_path)
                expected_frames = expected_pixel_frames(latent_frames)
                if decoded_frames != expected_frames:
                    raise RuntimeError(
                        f"decoded frame count {decoded_frames} != expected {expected_frames}"
                    )
                stats = pipeline.sparse_history_aggregate_stats.as_dict()
                route_digest, route_shas = _route_digest(stats)
                (case_dir / "sparse_history_stats.json").write_text(
                    json.dumps(stats, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                case_config = {
                    **identity,
                    "method": method,
                    "routing_stage": method_spec(method).routing_stage,
                    "backend": backend,
                    "history_density": history_density,
                    "refresh_policy": refresh_policy,
                    "rope_policy": rope_policy,
                    "latent_frames": latent_frames,
                    "prompt_id": prompt_id,
                    "prompt": case["prompt"],
                    "seed": seed,
                    "method_params": config.method_params,
                    "execution_commit": execution_commit,
                    "execution_change_scope": execution_change_scope,
                    "decode_mode": decode_mode,
                }
                (case_dir / "case_config.json").write_text(
                    json.dumps(case_config, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                state = {
                    **identity,
                    "method": method,
                    "status": "pass",
                    "routing_stage": method_spec(method).routing_stage,
                    "backend": backend,
                    "observed_attention_backend": stats.get("attention_backend"),
                    "history_density": history_density,
                    "refresh_policy": config.refresh_policy,
                    "rope_policy": config.rope_policy,
                    "latent_frames": latent_frames,
                    "route_plan_sha256": route_digest,
                    "route_plan_sha256s": route_shas,
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "video": str(video_path),
                    "video_sha256": _sha256(video_path),
                    "pixel_frames": int(frames.shape[1]),
                    "decoded_frames": decoded_frames,
                    "end_to_end_s": time.perf_counter() - started,
                    "model_load_s_total": model_load_s,
                    "model_load_s_amortized": load_amortized_s,
                    "peak_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
                    "history_pair_density": stats.get("history_pair_density"),
                    "history_transfer_density": stats.get("history_transfer_density"),
                    "global_executed_density": stats.get("global_executed_density"),
                    "candidate_transfer_bytes": stats.get("candidate_transfer_bytes"),
                    "transferred_bytes": stats.get("transferred_bytes"),
                    "archive_bytes": stats.get("archive_bytes"),
                    "index_bytes": stats.get("index_bytes"),
                    "index_transfer_bytes": stats.get("index_transfer_bytes"),
                    "query_summary_bytes": stats.get("query_summary_bytes"),
                    "staging_padding_tokens": stats.get("staging_padding_tokens"),
                    "selected_history_tokens": stats.get("selected_history_tokens"),
                    "candidate_history_tokens": stats.get("candidate_history_tokens"),
                    "executed_qk_pairs": stats.get("executed_qk_pairs"),
                    "dense_qk_pairs": stats.get("dense_qk_pairs"),
                    "attention_s": stats.get("timing", {}).get("attention_s"),
                    "routing_s": stats.get("timing", {}).get("routing_s"),
                    "q_summary_s": stats.get("timing", {}).get("q_summary_s"),
                    "d2h_s": stats.get("timing", {}).get("d2h_s"),
                    "cpu_gather_s": stats.get("timing", {}).get("cpu_gather_s"),
                    "h2d_s": stats.get("timing", {}).get("h2d_s"),
                    "rope_s": stats.get("timing", {}).get("rope_s"),
                    "stats": str(case_dir / "sparse_history_stats.json"),
                    "config": str(case_dir / "case_config.json"),
                    "failed_calls": stats.get("failed_calls", 0),
                    "fallback_calls": stats.get("dense_fallback_calls", 0),
                    "nan_calls": 0,
                    "execution_commit": execution_commit,
                    "execution_change_scope": execution_change_scope,
                    "decode_mode": decode_mode,
                }
                if state["failed_calls"] or state["fallback_calls"]:
                    raise RuntimeError("successful generation reported failed/fallback calls")
                state["end_to_end_with_amortized_load_s"] = (
                    state["end_to_end_s"] + load_amortized_s
                )
                state_path.write_text(
                    json.dumps(state, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                case_states.append(state)
            except Exception as error:
                state = {
                    **identity,
                    "method": method,
                    "status": "fail",
                    "routing_stage": method_spec(method).routing_stage,
                    "backend": backend,
                    "history_density": history_density,
                    "refresh_policy": config.refresh_policy,
                    "rope_policy": config.rope_policy,
                    "latent_frames": latent_frames,
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "end_to_end_s": time.perf_counter() - started,
                    "failure_reason": f"{type(error).__name__}: {error}",
                    "execution_commit": execution_commit,
                    "execution_change_scope": execution_change_scope,
                }
                (case_dir / "failure.json").write_text(
                    json.dumps(state, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                state_path.write_text(
                    json.dumps(state, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                case_states.append(state)
            finally:
                pipeline.vae.model.clear_cache()
                torch.cuda.empty_cache()
                gc.collect()
    (output_root / f"shard_{args.shard_index}_states.json").write_text(
        json.dumps(
            {
                "commit": commit,
                "model_load_s": model_load_s,
                "shard_axis": args.shard_axis,
                "cases": case_states,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if any(case["status"] == "fail" for case in case_states):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
