# LongLive2.0 official-source boundary

Official repository: https://github.com/NVlabs/LongLive
`main` resolved by live `git ls-remote` to
`00c44e5418e8a5333a5cb3871c54948d5935b950`.
Paper: https://arxiv.org/abs/2605.18739.
This is a source audit, not a reproduced LongLive2 result or a speed comparison.

The official README links v1.0 separately and describes LongLive2 as NVFP4/
parallel infrastructure with a Wan2.2-TI2V-5B model, multi-shot sinks, sequence
parallel inference and optional async/streaming VAE. Published45.7FPS is a
two-step NVFP4 setting; it must not be compared directly to our four-step1.3B
BF16 RAG/4090 timings. Model capacity, precision, hardware, steps, resolution,
frame-rate and VAE accounting differ.

Pinned `configs/inference.yaml`: eight latent frames/block, local attention32,
sink8, sampling_steps4, latent shape[1,128,48,44,80]; streaming/async VAE default
false. Its example `vae_device: cuda:2` is inactive unless streaming relocation
is enabled; any future single-GPU adaptation must bind devices explicitly.

Pinned `configs/fp8/inference_fp8.yaml` uses the merged BF16 checkpoint, TorchAO
row-wise FP8 W8A8, torch.compile and max-autotune-no-cudagraphs. The README reports
a validated PyTorch2.8/cu128 + TorchAO0.13 stack on H100, capability8.9+; the
provided example validates only one eight-latent block. Long shapes/compile
fallbacks require separate verification. These are upstream claims, not ours.

Research implications: keep a same-backbone BF16 algorithm/system comparison
separate from cross-model deployment comparisons; async VAE and quantized KV
are additional meaningful baselines. Our `upstream_zero` spatial prototypes
cannot be transplanted into a different RoPE/cache policy without proving that
their coordinates match execution. Do not cast NVFP4 pipelines with a generic
BF16 `.to(...)`, which the upstream README explicitly warns against.

No new weights were downloaded, shared environments changed, or training run.
Next baseline work must lock its own weights/runtime and define fair comparison
groups; current1.3B RAG results are not a claim to beat LongLive2.0.
