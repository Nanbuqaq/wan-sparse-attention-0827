#!/usr/bin/env python3
"""Strict one-step latent gate with Dense-repeat noise and three backends."""

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

from adapters import MethodConfig, SparseRunStats, install_sparse_processors


PROMPT = (
    "A macro slow-motion shot of a hummingbird hovering between red flowers, "
    "extremely rapid wing motion, drifting pollen, shallow depth of field, natural sunlight, highly detailed, cinematic, 4k"
)


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    first = reference.float().cpu()
    second = candidate.float().cpu()
    delta = second - first
    return {
        "shape": list(first.shape),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "rmse": float(torch.sqrt(torch.mean(delta * delta))),
        "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(first).clamp_min(1e-12)),
        "cosine": float(torch.nn.functional.cosine_similarity(first.flatten(), second.flatten(), dim=0)),
        "finite": bool(torch.isfinite(second).all()),
    }


def main() -> None:
    from diffusers import WanPipeline

    pipe = WanPipeline.from_pretrained(str(wan_model_path()), torch_dtype=torch.bfloat16)
    if hasattr(pipe.scheduler, "shift"):
        pipe.scheduler.shift = 8.0
    native = dict(pipe.transformer.attn_processors)
    pipe.enable_model_cpu_offload()

    def run(seed: int = 42):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = pipe(
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
        return output, time.perf_counter() - start

    pipe.transformer.set_attn_processor(dict(native))
    dense, dense_s = run()
    pipe.transformer.set_attn_processor(dict(native))
    dense_repeat, dense_repeat_s = run()
    dense_noise = metrics(dense, dense_repeat)
    payload = {
        "schema_version": 2,
        "dense_elapsed_s": dense_s,
        "dense_repeat_elapsed_s": dense_repeat_s,
        "dense_repeat_noise": dense_noise,
        "thresholds": {
            "relative_l2": max(0.01, 5.0 * dense_noise["relative_l2"]),
            "cosine": 0.9999,
        },
        "methods": {},
    }
    for method, backend, backend_params, q_clusters, k_clusters in (
        ("original_block", "fixed64_bf16", {}, 1, 1),
        ("svg2", "fixed64_bf16", {}, 300, 1000),
        ("svg2", "varlen_triton_native", {}, 300, 1000),
        ("svg2", "varlen_triton_csr", {"block_m": 64, "block_n": 32}, 300, 1000),
    ):
        pipe.transformer.set_attn_processor(dict(native))
        stats = SparseRunStats()
        install_sparse_processors(
            pipe.transformer,
            config=MethodConfig(
                method=method,
                backend=backend,
                density=1.0,
                parameter_origin="q300k1000_strict_equivalence_v2",
                q_clusters=q_clusters,
                k_clusters=k_clusters,
                kmeans_init_iterations=10,
                kmeans_step_iterations=2,
                inference_steps=1,
                backend_params=backend_params,
            ),
            stats=stats,
        )
        candidate, elapsed_s = run()
        key = f"{method}:{backend}"
        latent = metrics(dense, candidate)
        passed = (
            latent["finite"]
            and latent["relative_l2"] <= payload["thresholds"]["relative_l2"]
            and latent["cosine"] >= payload["thresholds"]["cosine"]
            and stats.failed_calls == 0
            and stats.dense_fallback_calls == 0
        )
        payload["methods"][key] = {
            "elapsed_s": elapsed_s,
            "latent": latent,
            "sparse": stats.as_dict(),
            "status": "pass" if passed else "fail",
        }
    payload["status"] = "pass" if all(
        item["status"] == "pass"
        for item in payload["methods"].values()
    ) else "fail"
    output = ROOT / "results" / "metrics" / "latent_equivalence_v2.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
