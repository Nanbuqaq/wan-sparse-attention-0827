# LongLive2 strong-baseline audit (source only, not measured performance)

- Official repository: https://github.com/NVlabs/LongLive
- Read-only source lock: `6b36d20ec6f7958d29d11a704dfa64611a9f2572`, verified via connected GitHub commit API during this sprint.
- Paper: **LongLive2.0: An NVFP4 Parallel Infrastructure for Long Video Generation**, arXiv2605.18739.
- [Pinned README](https://github.com/NVlabs/LongLive/blob/6b36d20ec6f7958d29d11a704dfa64611a9f2572/README.md): BF16, W4A4/NVFP4 KV, FP8 PTQ, multi-shot sink, sequence parallel inference, asynchronous decoding.
- [Pinned default inference config](https://github.com/NVlabs/LongLive/blob/6b36d20ec6f7958d29d11a704dfa64611a9f2572/configs/inference.yaml): Wan2.2-TI2V-5B, 8-latent blocks, local attention32, sink8, four sampling steps; latent shape `[1,128,48,44,80]`. Streaming/async VAE are available but false in this default config.
- Published BF16 checkpoint: `Efficient-Large-Model/LongLive-2.0-5B/model_bf16.pt`; README says remove the adapter section for full-generator inference. Do not accidentally attach our1.3B LoRA.
- FP8 documented validation uses Python3.10, Torch2.8.0+cu128, TorchAO0.13.0 on H100. The README explicitly limits the supplied validated FP8 config to one8-latent block and warns of extra shape compilation/eager fallback at longer lengths.
- Its reported throughput is **not** a directly comparable result against our1.3B/BF16/3-latent/480×832 RAG pipeline. Hardware, backbone, resolution, steps and memory context differ.
- It already includes fused RoPE/adaLN, reduced cache synchronization, in-place quantized cache updates and pinned VAE transfers. Generic implementations of those are baseline engineering, not automatically our novel contribution.
- Research implication: establish same-backbone method/system ablations first, then a separately reported native LongLive2 baseline and transferability check. Information-group/lifetime admission and structured long-history access should be the potential contribution, not claiming ownership of generic async/quantization ideas.
- This audit has not downloaded5B weights, installed its environment, or run its inference. Those steps must remain bounded by research value, not turn into another exhaustive reproduction effort.
