#!/usr/bin/env python3
"""Test the released VAE's reset behavior using an already-generated latent."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.video_decode import decode_latents_chunked_exact
from adapters.longlive_sparse.offline_eval import output_error_metrics


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--latent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("do not overwrite VAE continuity evidence")
    os.environ["MODEL_ROOT"] = str(args.model_root)
    sys.path.insert(0, str(args.source))
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.enabled = False
    from utils.wan_wrapper import WanVAEWrapper
    latent = torch.load(args.latent, map_location="cpu", weights_only=True).cuda()
    vae = WanVAEWrapper().eval().requires_grad_(False).to(device="cuda", dtype=torch.bfloat16)
    original = vae.decode_to_pixel_chunk(latent, use_cache=False, chunk_size=60)
    reset = vae.decode_to_pixel_chunk(latent, use_cache=False, chunk_size=3)
    continuous = decode_latents_chunked_exact(vae, latent, chunk_size=3)
    torch.cuda.synchronize()
    error = output_error_metrics(original, continuous)
    passed = torch.equal(original, continuous)
    result = {"status": "pass" if passed else "fail", "scope": "official_source_VAE_only_no_generation_rerun",
        "latent_frames": int(latent.shape[1]), "released_chunk60_frames": int(original.shape[1]),
        "released_forced_chunk3_frames": int(reset.shape[1]), "continuous_chunk3_frames": int(continuous.shape[1]),
        "continuous_exact_one_shot": passed, "numerical_error": error,
        "expected_default60_frames_from_source_rule_not_long_video_measurement": {"120": 474, "240": 948},
        "correct_continuous_expected_frames": {"120": 477, "240": 957},
        "vae_wrapper_sha256": hashlib.sha256((args.source / "utils/wan_wrapper.py").read_bytes()).hexdigest(),
        "gpu": torch.cuda.get_device_name(), "cudnn_enabled": torch.backends.cudnn.enabled}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
