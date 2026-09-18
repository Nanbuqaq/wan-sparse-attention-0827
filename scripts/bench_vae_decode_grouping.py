#!/usr/bin/env python3
"""Isolated Wan2.2 VAE decode benchmark: per-latent loop vs grouped-latent calls.

Fixed real latents (from a completed InferHub case) are decoded by
(1) the official one-latent-at-a-time cached loop (current streaming adapter
semantics) and (2) grouped calls passing k latents per Decoder3d invocation.
Grouped decode reuses the same feat_cache mechanism; causality is preserved
because CausalConv3d/Resample only ever read past frames.

Reports CUDA-event service time per variant and the pixel-space drift of each
grouped variant vs the per-latent reference. No weights are modified.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
LONG_LIVE = ROOT / "third_party" / "LongLive2"
sys.path.insert(0, str(LONG_LIVE))

from wan_5b.modules.vae2_2 import WanVAE_, unpatchify  # noqa: E402

MEAN = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
]
STD = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
]



def load_vae(path: Path, device, dtype):
    model = WanVAE_(dim=160, dec_dim=256, z_dim=48, dim_mult=[1, 2, 4, 4],
                    num_res_blocks=2, attn_scales=[], temperal_downsample=[False, True, True])
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model = model.eval().requires_grad_(False).to(device=device, dtype=dtype)
    model.clear_cache()
    return model


def normalize(model, z, mean, std):
    return model.conv2(z / std.view(1, model.z_dim, 1, 1, 1) + mean.view(1, model.z_dim, 1, 1, 1))


def decode_per_latent(model, x, unpatchify_fn, decoder=None):
    """Official cached loop: one latent per decoder call (reference)."""
    decoder = decoder or model.decoder
    outs = []
    model.clear_cache()
    for index in range(x.shape[2]):
        model._conv_idx = [0]
        kwargs = dict(feat_cache=model._feat_map, feat_idx=model._conv_idx)
        if index == 0:
            kwargs["first_chunk"] = True
        outs.append(unpatchify_fn(decoder(x[:, :, index:index + 1], **kwargs), patch_size=2))
    model.clear_cache()
    return torch.cat(outs, dim=2)


def decode_grouped(model, x, group, unpatchify_fn):
    """k latents per decoder call; first latent stays a first_chunk T=1 call."""
    outs = []
    model.clear_cache()
    index = 0
    while index < x.shape[2]:
        size = 1 if index == 0 else min(group, x.shape[2] - index)
        model._conv_idx = [0]
        kwargs = dict(feat_cache=model._feat_map, feat_idx=model._conv_idx)
        if index == 0:
            kwargs["first_chunk"] = True
        outs.append(unpatchify_fn(model.decoder(x[:, :, index:index + size], **kwargs), patch_size=2))
        index += size
    model.clear_cache()
    return torch.cat(outs, dim=2)


def timed(fn, warmup=1, repeat=3):
    for _ in range(warmup):
        fn()
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    spans = []
    out = None
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        spans.append(start.elapsed_time(end) / 1000.0)
        torch.cuda.empty_cache()
    return out, min(spans)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--latents", type=Path, required=True, help="BTCHW latent tensor from a real case")
    parser.add_argument("--latent-count", type=int, default=17)
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--compile-modes", nargs="*", default=[],
                        choices=["default", "max-autotune-no-cudagraphs"],
                        help="also benchmark torch.compile'd decoder variants")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    model = load_vae(args.vae, device, dtype)
    mean = torch.tensor(MEAN, dtype=dtype, device=device)
    std = torch.tensor(STD, dtype=dtype, device=device)

    latents_btchw = torch.load(args.latents, map_location="cpu")
    if latents_btchw.ndim != 5:
        raise ValueError("expected 5D BTCHW latents")
    # BTCHW -> BCTHW for the VAE, take a fixed prefix.
    z = latents_btchw[:, :args.latent_count].permute(0, 2, 1, 3, 4).to(device=device, dtype=dtype)

    result = {"vae": str(args.vae), "latents": str(args.latents),
              "latent_prefix": int(z.shape[2]), "device": torch.cuda.get_device_name(device),
              "dtype": str(dtype), "variants": {}}
    with torch.inference_mode():
        x = normalize(model, z, mean, std)
        del z
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        ref_pixels, ref_s = timed(lambda: decode_per_latent(model, x, unpatchify))
        result["variants"]["per_latent_official"] = {
            "decode_service_s": ref_s,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
        ref = ref_pixels.float().cpu()
        del ref_pixels
        torch.cuda.empty_cache()
        for group in args.group_sizes:
            torch.cuda.reset_peak_memory_stats(device)
            pixels, service_s = timed(lambda: decode_grouped(model, x, group, unpatchify))
            diff = (pixels.float().cpu() - ref).abs()
            rel_l2 = (diff.pow(2).sum().sqrt() / ref.pow(2).sum().sqrt()).item()
            result["variants"][f"grouped_{group}"] = {
                "decode_service_s": service_s,
                "speedup_vs_official": ref_s / service_s,
                "max_abs_diff": diff.max().item(),
                "relative_l2": rel_l2,
                "pixel_frames": int(pixels.shape[2]),
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
            del pixels, diff
            torch.cuda.empty_cache()
        for mode in args.compile_modes:
            compile_kwargs = {} if mode == "default" else {"mode": mode}
            compiled = torch.compile(model.decoder, dynamic=False, **compile_kwargs)
            torch.cuda.reset_peak_memory_stats(device)
            pixels, service_s = timed(lambda: decode_per_latent(model, x, unpatchify, decoder=compiled))
            diff = (pixels.float().cpu() - ref).abs()
            rel_l2 = (diff.pow(2).sum().sqrt() / ref.pow(2).sum().sqrt()).item()
            result["variants"][f"compiled_{mode}"] = {
                "decode_service_s": service_s,
                "speedup_vs_official": ref_s / service_s,
                "max_abs_diff": diff.max().item(),
                "relative_l2": rel_l2,
                "pixel_frames": int(pixels.shape[2]),
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
            del pixels, diff
            torch.cuda.empty_cache()
        result["pixel_frames"] = int(ref.shape[2])
        result["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
