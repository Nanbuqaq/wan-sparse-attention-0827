#!/usr/bin/env python3
"""Explicit startup -> text -> generation -> VAE -> RGB -> MP4 diagnostic.

This is an instrumented workflow, not a production speed measurement. It keeps
an uninstrumented same-seed latent control and preserves every failure artifact.
Pinned upstream source is read-only; all instrumentation is process-local.
"""
import time
PROCESS_STARTED = time.perf_counter()
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.full_flow_profile import FullFlowTrace, instrument_pipeline, pipeline_regions, normalize_raw_vae, unit_video_to_rgb
from adapters.longlive_sparse.history_cache import tensor_sha256
from adapters.longlive_sparse.system_config import LongLiveSystemConfig
from adapters.longlive_sparse.video_decode import decode_latents_chunked_exact


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("rag_dense", "transfer_vaware_hybrid_history"), required=True)
    parser.add_argument("--latent-frames", type=int, default=39)
    parser.add_argument("--prompt-id", default="calibration_motion")
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--count-work", action="store_true",help="record actual module shapes and full plan arithmetic (diagnostic only)")
    args = parser.parse_args()
    if args.latent_frames < 21 or args.latent_frames % 3:
        parser.error("use a block-aligned trajectory exercising history")
    prompt = next(p for p in json.loads((ROOT / "configs/system/profile_calibration_prompts.json").read_text())["candidates"]
                  if p["prompt_id"] == args.prompt_id)
    if args.validate_only:
        print(json.dumps({"status": "validated", "method": args.method, "latent_frames": args.latent_frames,
                          "pixel_frames": 4 * args.latent_frames - 3, "prompt_id": args.prompt_id}))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA required")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    os.environ.update(INFER_OUTPUT_DIR=str(root), LONGLIVE_NVTX="1" if args.nvtx else "0",
                      LONGLIVE_CAPTURE_QKV="0", LONGLIVE_CAPTURE_COMPLETE_ATTENTION="0")
    config = yaml.safe_load((ROOT / "configs/inferhub/rag_method_21.yaml").read_text())
    empty = root / "empty_prompts.txt"
    empty.write_text("")
    config.update(data_path=str(empty), output_folder=str(root / "load"), inference_iter=0)
    params = json.loads((ROOT / "configs/formal/method_params.json").read_text())["method_params"].get(args.method, {})
    config["sparse_history"].update(method=args.method, history_density=1. if args.method == "rag_dense" else .25,
                                   method_params=params, record_per_call=True)
    system = LongLiveSystemConfig(profile_mode="trace", transfer_layout="exact_compact", staging_mode="persistent_separate",
        cpu_pack_policy="archive_runs", gpu_union_cache="per_chunk", gpu_union_cache_budget_mib=4096,
        archive_offload="pooled_pageable", host_pinned_budget_mib=128)
    config["longlive_system"] = system.as_dict()
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    trace = FullFlowTrace(cuda=True, nvtx=args.nvtx)
    report = {"status": "running", "scope": "instrumented_full_flow_with_latent_control",
        "method": args.method, "latent_frames": args.latent_frames, "pixel_frames": 4 * args.latent_frames - 3,
        "prompt_id": args.prompt_id, "prompt_sha256": hashlib.sha256(prompt["prompt"].encode()).hexdigest(), "seed": args.seed,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "system": system.as_dict(),
        "initial_import_wall_s": time.perf_counter() - PROCESS_STARTED,
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "instrumentation_sha256": digest(ROOT / "adapters/longlive_sparse/full_flow_profile.py"),
        "driver_sha256": digest(__file__), "config_sha256": digest(config_path)}
    (root / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
    work_meter=None
    try:
        trace.wrap(torch, "load", "startup.torch_load", device="CPU/storage", cuda=False,
                   metadata=lambda a, kw: {"file": str(a[0] if a else kw.get("f"))})
        trace.wrap(torch.nn.Module, "load_state_dict", "startup.load_state_dict", device="CPU/GPU")
        trace.wrap(torch.nn.Module, "to", "startup.module_to", device="CPU/GPU")
        with trace.span("startup.load_pipeline"):
            from scripts.run_longlive_sparse import run_config
            pipeline = run_config(config_path)["pipeline"]
            torch.cuda.synchronize()
        trace.restore()
        print("FULL_FLOW_MODEL_LOADED", flush=True)
        from utils.misc import set_seed
        from adapters.longlive_sparse.upstreams import load_rag_pipeline_module
        rag_module = load_rag_pipeline_module()
        original_function = rag_module.CausalInferencePipeline.inference
        instrument_pipeline(trace, pipeline)
        work_meter=None
        if args.count_work:
            from adapters.longlive_sparse.operator_work import OperatorWorkMeter
            work_meter=OperatorWorkMeter();work_meter.attach(pipeline)
        trace.wrap(rag_module, "move_model_to_device_with_memory_preservation", "text.post_encode_residency")
        set_seed(args.seed)
        with trace.span("input.noise", device="GPU"):
            noise = torch.randn(1, args.latent_frames, 16, 60, 104, device="cuda", dtype=torch.bfloat16)
        report["initial_noise_sha256"] = tensor_sha256(noise)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), trace.span("generation.complete"):
            with pipeline_regions(trace, original_function):
                _, latent = pipeline.inference(noise=noise, text_prompts=[prompt["prompt"]], return_latents=True,
                                               low_memory=True, profile=False, skip_vae_decode=True)
            torch.cuda.synchronize()
        report["generation_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["archive"] = pipeline.sparse_history_archive.storage_summary()
        report["component_service_timers_not_additive"] = pipeline.sparse_history_aggregate_stats.as_dict()["timing"]
        report["history_cache"] = pipeline.history_union_cache.as_dict()
        with trace.span("validation.latent_and_hash", device="CPU+GPU"):
            assert torch.isfinite(latent).all()
            report["latent_sha256"] = tensor_sha256(latent)
        with trace.span("vae.decode_complete", device="CPU+GPU"), torch.inference_mode():
            video = decode_latents_chunked_exact(pipeline.vae, latent.cuda(), chunk_size=120)
            torch.cuda.synchronize()
        with trace.span("output.normalize_VAE_CPU", device="CPU", cuda=False):
            video = normalize_raw_vae(video)
        with trace.span("output.RGB_convert_CPU", device="CPU", cuda=False):
            pixels = unit_video_to_rgb(video)[0]
        with trace.span("output.latent_save", device="CPU/storage", cuda=False):
            torch.save(latent.cpu(), root / "latents.pt")
        with trace.span("output.MP4_encode_write", device="CPU/storage", cuda=False):
            from torchvision.io import write_video
            write_video(str(root / "video.mp4"), pixels, fps=16)
        with trace.span("output.decode_integrity_check", device="CPU/storage", cuda=False):
            import av
            with av.open(str(root / "video.mp4")) as container:
                frames = sum(1 for _ in container.decode(video=0))
            assert frames == report["pixel_frames"]
            report["video_sha256"] = digest(root / "video.mp4")
        if work_meter is not None:
            work_meter.detach()
            full_stats=pipeline.sparse_history_aggregate_stats.as_dict()
            head_dim=int(pipeline.generator.model.blocks[0].self_attn.head_dim)
            report['operator_work']=work_meter.result(full_stats,head_dim)
            (root/'generation_call_stats.json').write_text(json.dumps(full_stats,indent=2)+'\n')
            (root/'operator_work.json').write_text(json.dumps(report['operator_work'],indent=2)+'\n')
        trace.restore()
        report["trace"] = trace.result()
        print("FULL_FLOW_PROFILE_COMPLETE; running uninstrumented latent control", flush=True)
        # Same inference API and seed; profile-only changes must preserve latents.
        del video, pixels, noise
        os.environ["LONGLIVE_NVTX"] = "0"
        set_seed(args.seed)
        noise = torch.randn(1, args.latent_frames, 16, 60, 104, device="cuda", dtype=torch.bfloat16)
        control_start = time.perf_counter()
        with torch.inference_mode():
            _, control = pipeline.inference(noise=noise, text_prompts=[prompt["prompt"]], return_latents=True,
                                            low_memory=True, profile=False, skip_vae_decode=True)
        torch.cuda.synchronize()
        report["control_generation_wall_s"] = time.perf_counter() - control_start
        report["control_latent_sha256"] = tensor_sha256(control)
        report["instrumented_latent_exact_control"] = torch.equal(latent.cpu(), control.cpu())
        if not report["instrumented_latent_exact_control"]:
            raise RuntimeError("profile changes latent trajectory; do not use timings")
        report["status"] = "pass"
        report["preview_protocol"] = "VAE_raw_minus1_plus1_to_unit_to_uint8_v2"
    except BaseException:
        report["status"] = "fail"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        if work_meter is not None:
            work_meter.detach()
        trace.restore()
        if not trace.stack:
            report["trace"] = trace.result()
        report["process_wall_s_including_control"] = time.perf_counter() - PROCESS_STARTED
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
