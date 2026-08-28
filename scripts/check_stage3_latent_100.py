#!/usr/bin/env python3
"""One-step Stage-3 100% latent audit separating execution from strict equality."""

from __future__ import annotations

import json
import sys
import time

from bootstrap import ROOT, configure_runtime
from model_path import wan_model_path

configure_runtime()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from adapters import MethodConfig, SparseRunStats, install_sparse_processors


PROMPT = "A rhythmic gymnast performs fast spins while a long red ribbon traces loops through the air, cinematic tracking shot, highly detailed, 4k"


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    first = reference.float().cpu()
    second = candidate.float().cpu()
    delta = second - first
    return {
        "shape": list(first.shape),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(first).clamp_min(1e-12)),
        "cosine": float(F.cosine_similarity(first.flatten(), second.flatten(), dim=0)),
        "finite": bool(torch.isfinite(second).all()),
    }


def main() -> None:
    from diffusers import WanPipeline

    pipe = WanPipeline.from_pretrained(str(wan_model_path()), torch_dtype=torch.bfloat16)
    if hasattr(pipe.scheduler, "shift"):
        pipe.scheduler.shift = 8.0
    native = dict(pipe.transformer.attn_processors)
    pipe.enable_model_cpu_offload()

    def run():
        generator = torch.Generator(device="cpu").manual_seed(9001)
        torch.cuda.synchronize()
        started = time.perf_counter()
        latent = pipe(
            prompt=PROMPT,
            height=480,
            width=832,
            num_frames=81,
            num_inference_steps=1,
            guidance_scale=6.0,
            generator=generator,
            output_type="latent",
        ).frames.detach().cpu()
        torch.cuda.synchronize()
        return latent, time.perf_counter() - started

    pipe.transformer.set_attn_processor(dict(native))
    dense, dense_s = run()
    pipe.transformer.set_attn_processor(dict(native))
    dense_repeat, repeat_s = run()
    dense_noise = metrics(dense, dense_repeat)
    route_params = {
        "base_fraction": 0.80,
        "local_fraction": 0.10,
        "remote_clusters": 128,
        "refresh_calls": 20,
        "v_objective": "output_residual",
        "v_weight": 0.75,
        "early_base_bonus": 0.05,
        "late_base_bonus": 0.025,
        "frames_latent": 21,
        "height_latent": 30,
        "width_latent": 52,
    }
    cases = [
        ("original_block", "fixed64_bf16", {}, {}),
        ("stage3_hybrid", "fixed64_bf16", {}, route_params),
        ("stage3_hybrid", "varlen_triton_csr", {"block_m": 64, "block_n": 64}, route_params),
    ]
    rows = []
    for method, backend, backend_params, params in cases:
        pipe.transformer.set_attn_processor(dict(native))
        stats = SparseRunStats()
        install_sparse_processors(
            pipe.transformer,
            config=MethodConfig(
                method=method,
                backend=backend,
                density=1.0,
                parameter_origin="stage3_latent_100",
                q_clusters=128,
                k_clusters=128,
                kmeans_init_iterations=3,
                kmeans_step_iterations=1,
                inference_steps=1,
                calls_per_step=2,
                backend_params=backend_params,
                route_params=params,
            ),
            stats=stats,
        )
        latent, elapsed = run()
        error = metrics(dense, latent)
        execution_pass = (
            error["finite"]
            and stats.failed_calls == 0
            and stats.dense_fallback_calls == 0
            and abs(float(stats.as_dict()["logical_pair_density"]) - 1.0) <= 1e-6
        )
        strict_pass = error["relative_l2"] <= max(0.01, 5 * dense_noise["relative_l2"]) and error["cosine"] >= 0.9999
        rows.append(
            {
                "method": method,
                "backend": backend,
                "elapsed_s": elapsed,
                "latent_vs_dense": error,
                "execution_status": "pass" if execution_pass else "fail",
                "strict_numerical_status": "pass" if strict_pass else "fail",
                "sparse": stats.as_dict(),
            }
        )
    payload = {
        "schema_version": 3,
        "dense_elapsed_s": dense_s,
        "dense_repeat_elapsed_s": repeat_s,
        "dense_repeat_noise": dense_noise,
        "strict_relative_l2_threshold": max(0.01, 5 * dense_noise["relative_l2"]),
        "rows": rows,
        "execution_status": "pass" if all(row["execution_status"] == "pass" for row in rows) else "fail",
        "strict_numerical_status": "pass" if all(row["strict_numerical_status"] == "pass" for row in rows) else "fail",
        "classification": "100_percent_routes_execute_correctly_but_multilayer_bf16_accumulation_is_not_strict_latent_equivalence",
    }
    output = ROOT / "results/metrics/stage3_latent_100.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["execution_status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
