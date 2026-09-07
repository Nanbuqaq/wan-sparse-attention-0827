#!/usr/bin/env python3
"""Source-weight gate audit and descriptive storyboards, not a paper ranking."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.offline_eval import output_error_metrics
from scripts.build_video_review_storyboards import analyze


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", choices=("gate21", "group"), default="gate21")
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--latent-frames", type=int, default=21)
    parser.add_argument("--prompt-id", default="official_teapot")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    paths = {"reference": args.case / "reference_longlive_rag.mp4", "tether": args.case / "tether_recovery_v3/tethermem.mp4",
             "neutral_split": args.case / "neutral_gate21/video.mp4"}
    if args.layout == "group":
        paths.update(tether=args.case / "tethermem.mp4", neutral_split=args.case / "neutral_split_sdpa/video.mp4")
    cases, latents, errors = [], {}, {}
    for name, video in paths.items():
        run = json.loads(video.with_suffix(".run.json").read_text())
        if run["status"] != "completed" or run["seed"] != args.seed or run["latent_frames"] != args.latent_frames:
            raise ValueError("unmatched or incomplete gate video")
        latent_file = video.with_suffix(".latents.pt")
        latents[name] = torch.load(latent_file, map_location="cpu", weights_only=True)
        case = {"id": name, "method": name, "status": "pass", "prompt_id": args.prompt_id, "seed": args.seed,
                "latent_frames": args.latent_frames, "video": str(video), "video_sha256": sha(video), "latent_file_sha256": sha(latent_file)}
        cases.append(case)
        analyze(case, args.output, samples_per_quarter=16)
    for name in ("tether", "neutral_split"):
        errors[name] = {"latent_exact_reference": torch.equal(latents["reference"], latents[name]),
                        "all_latent": output_error_metrics(latents["reference"], latents[name]),
                        "last_chunk_latent": output_error_metrics(latents["reference"][:, -3:], latents[name][:, -3:])}
        errors[name]["quarters"] = [output_error_metrics(latents["reference"][:, a:b], latents[name][:, a:b])
            for a, b in zip([round(i*args.latent_frames/4) for i in range(4)], [round(i*args.latent_frames/4) for i in range(1, 5)])]
    result = {"status": "pass", "cases": cases, "latent_errors_not_absolute_quality": errors,
              "scope": "source_weight_matched_group_diagnostics_not_absolute_quality_ranking",
              "backbone": "causal_forcing_plus_AE_no_LoRA", "old_LoRA_avgpool_results_not_pooled": True,
              "review_is_not_blind_human_preference": True, "initial_noise_SHA_not_recorded_in_all_early_gate_variants": True}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(errors, indent=2))


if __name__ == "__main__":
    main()
