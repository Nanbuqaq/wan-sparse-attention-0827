# LongLive2 native BF16 reference — feasibility and fixed scope

## Why this control

The current1.3B absence/reappearance stress has transition failures even with
official interactive recache. A newer native backbone is a capability reference,
not a same-backbone ablation or a basis for comparing absoluteFPS. No5B sparse
method port, training, quantization sweep or perfect replication is required.

## Source and weights

- Official `NVlabs/LongLive` source6b36d20ec6f7958d29d11a704dfa64611a9f2572;
  separately pinned Git submodule `third_party/LongLive2`.
- Official full generator `Efficient-Large-Model/LongLive-2.0-5B`, revision
  8521079b863720a57c1a8d9b19c8d9e6ccb04c0f; `model_bf16.pt`9999853030bytes,
  SHA256ec9063a44ea3c91e8ff55edcdd58dba3f1bcf6ac9091249629cb57fcebe35fd8.
- Wan2.2-TI2V-5B assets revision921dbaf3f1674a56f47e83fb80a34bac8a8f203e.
  VAE2818839170bytes, SHA25620eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36.
- Existing T5 is reused only after verifying its exact official SHA256
  7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d.
  All downloads and dependency work remain in the private task environment.

## Loading and evaluation differences, declared in advance

Use official architecture, scheduler, Attention and inference loop unchanged.
The usual base20GB generator is entirely overwritten by the released full
generator. Initialize `CausalWanModel.from_config` in a scoped loading patch,
then use official `load_generator_checkpoint(strict=True)`; any missing or
unexpected key stops execution. No random or base parameter may remain. Remove
the default adapter and LoRA configuration. This is not original cold-start
timing, and no cold-start speedup will be claimed.

Encode each unique prompt using the native BF16 T5, preserve exact resulting
per-prompt embeddings, then offload T5. Use the native `return_latents=True`
path; after generation, free native KV and offload the generator before native
batch VAE decode. This staged placement allows a24GB local branch gate but is
not the official optimized multi-GPU serving configuration. Report each stage
and do not turn its duration into a cross-model speed comparison.

Native resolution: latent48×44×80, output704×1280,8-latent blocks. Native default
local32/sink8. The source currently allocates both positive and negative BF16
KV even when guidance=1, so the local gate explicitly uses local16 and24latent
frames/93pixels; it exercises multiple prompts and cache roll, not final quality.
The128latent/509pixel reference uses local32 and the original resolution.
Pixel count follows4F−3 (128→509), not511.

The toy stress schedule is resampled to block-aligned boundaries0/24/48/80.
Original1.3B used0/21/48/78 and120latents. Their outputs are descriptive
cross-backbone capability evidence, NOT a matched quality or latency experiment.

Conditional full reference: original two development seeds20260909/20260910 and
duck/absence controls if the real GPU gate passes and remaining sprint time
permits. Keep own-reveal identity, away compliance and replacement/removal as
separate questions. If resources/setup fail, report feasibility limits without
inventing native inference results.

## Prior art limits

The source already includes fused RoPE/adaLN, in-place KV updates, reduced sync,
and streaming/async VAE. These cannot be presented as new LongLive2-relative
contributions from our1.3B experiments. Optional FP8/NVFP4/FA4 backends are outside
this BF16 capability check.
