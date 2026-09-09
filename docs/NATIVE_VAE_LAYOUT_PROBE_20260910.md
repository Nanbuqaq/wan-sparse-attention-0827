# Profile-motivated decoder layout probe (not yet a GPU result)

The completed native two-device trace places about77.21s of kernel service on
the VAE/pixel GPU. Named nchwToNhwc/nhwcToNchw kernels account for about6.41s,
and one large direct-copy kernel family another6.69s. Those families are
observations, not all DRAM transactions or a promise that all can be removed.

This motivates a bounded memory-format probe, not a new model or weight change.
Compare baseline, convolution-weights-only channels-last, conv2-entry layout,
and all-convolution-input layout. Conv3d uses channels_last_3d and Conv2d uses
channels_last; preserve logical shapes, weight values, native decoder/cache
semantics and BF16. Restore formats/hooks after each independent video.

First use16 saved latents at native resolution, not new DiT video generation.
Check baseline native batch against original incremental decode, then each
variant against that reference: full float errors, raw RGB equality, finite
values, frame continuity and memory peaks. Include conversion/hook setup and
estimated input-reformat payload separately. Diagnostic runs with CPU hashing
are not latency benchmarks.

Only exact-output candidates can initially claim a lossless system path.
Nonexact variants retain their errors as approximate renderer candidates, with
no automatic promotion. Full128-latent equivalence and repeated complete-cost
measurement are required before adoption. If conversions remain or their
replacement costs dominate, preserve the negative. No FA4-specific instructions
are assumed available on the 4090, and no third-party source or weight file is edited.

The independent object-state batch runs first; do not take its locked GPUs.
