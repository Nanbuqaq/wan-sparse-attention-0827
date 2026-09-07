#!/usr/bin/env python3
"""Re-decode diagnostic latents, never rerun generation or overwrite previews."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from torchvision.io import write_video

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.full_flow_profile import normalize_raw_vae, unit_video_to_rgb
from adapters.longlive_sparse.history_cache import tensor_sha256
from adapters.longlive_sparse.video_decode import decode_latents_chunked_exact


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, os.environ["LONGLIVE_BASE_SOURCE"])
    from utils.wan_wrapper import WanVAEWrapper, _wan_model_dir
    weight = Path(_wan_model_dir()) / "Wan2.1_VAE.pth"
    weight_sha = sha(weight)
    if weight_sha != "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981":
        raise ValueError("unexpected VAE weights")
    vae = WanVAEWrapper().eval().requires_grad_(False).to(device="cuda", dtype=torch.bfloat16)
    for name in ("final39", "final120_v2", "dense120_nsys_v2", "final240_v2"):
        directory = args.root / name
        original = json.loads((directory / "report.json").read_text())
        if original["status"] != "pass" or not original["instrumented_latent_exact_control"]:
            raise ValueError("generation gate is not complete")
        output = directory / "preview_normalized_v2"
        output.mkdir(exist_ok=False)
        latent = torch.load(directory / "latents.pt", map_location="cpu", weights_only=True)
        if tensor_sha256(latent) != original["latent_sha256"]:
            raise ValueError("saved diagnostic latent changed")
        started = time.perf_counter()
        raw = decode_latents_chunked_exact(vae, latent.cuda(), chunk_size=120)
        normalized = normalize_raw_vae(raw)
        rgb = unit_video_to_rgb(normalized)[0]
        assert len(rgb) == 4 * original["latent_frames"] - 3
        # Independently match the already-used canonical renderer's formula.
        canonical = ((raw*.5+.5).clamp(0, 1)*255).to(torch.uint8).permute(0, 1, 3, 4, 2).contiguous()[0]
        if not torch.equal(rgb, canonical):
            raise ValueError("RGB differs from canonical evaluator")
        write_video(str(output / "video.mp4"), rgb, fps=16)
        import av
        with av.open(str(output / "video.mp4")) as container:
            decoded = sum(1 for _ in container.decode(video=0))
        if decoded != original["pixel_frames"]:
            raise ValueError("wrong encoded frame count")
        record = {"status": "pass", "latent_sha256": original["latent_sha256"], "raw_rgb_sha256": tensor_sha256(rgb),
            "original_report_sha256": sha(directory / "report.json"), "vae_sha256": weight_sha,
            "video_sha256": sha(output / "video.mp4"), "frames": decoded,
            "RGB_exact_canonical_formula": True, "generator_rerun": False,
            "recovery_wall_s_not_generation_latency": time.perf_counter() - started,
            "scope": "repair_diagnostic_preview_only_no_method_quality_ranking"}
        (output / "audit.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"case": name, **record}), flush=True)
        del latent, raw, normalized, rgb, canonical


if __name__ == "__main__":
    main()
