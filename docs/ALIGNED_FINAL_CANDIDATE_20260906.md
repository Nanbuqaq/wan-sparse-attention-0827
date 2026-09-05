# Causal RoPE-aligned Final: isolated runtime candidate

Motivation source: `cc8f2e9` clean-context 2x2, both development prompts, layers0/19,
latent30/114. Aligned-first improved full teacher output error on all eight
points, while current-vs-first refresh did not have a consistent advantage.
This is NOT full-denoising or video quality evidence and is not a novelty claim.

Frozen candidate ID: `rope_aligned_final_history`. Keep Final's 70/15/15 tiers,
V weight1, transfer multiplier1, 25% union, Block64 and per-chunk refresh in the
development suite. The legacy method ID and its raw-prototype path are unchanged.

At archive insertion, construct spatial-RoPE0 K means from raw K already on the
GPU. Keep original raw CPU K/V for actual transfers and re-RoPE execution. At
route refresh, summarize the already available post-RoPE Q on GPU. Reject raw-Q
summaries and unvalidated RoPE policies explicitly. Do not add current full KV,
teacher output, future video or learned external models to the online selector.

Only `upstream_zero` is supported initially. Frequency construction/rotation
and synchronization are charged to index time and the complete runtime. The
frequency table is cached by head dimension/device and cleared between runs;
no second full-KV archive is retained. Original full teacher evaluators reject
aligned prototypes rather than silently labeling them a raw legacy baseline.

Before any new candidate video: CPU regression plus real small/large five-call
forward/forced-eviction gates. Compare its chosen coordinates to an independent
preindexed phase reference, BF16 Attention to FP32 on identical original KV,
and all archived raw KV and newly evicted phase prototypes to canonical values.
Require four cache hits/one miss. `9e16f23` passed small/large real gates, with
worst BF16-vs-FP32 max-abs0.001389 and relative-L2 below0.00224; raw KV and new
phase prototypes match exactly, and both caches record four hits/one miss.

The video follow-up uses SIX 39-latent development cases: Dense/Final/aligned
on motion/state, three methods on the same physical GPU per prompt. The old
repaired H-pool controls are retained but not mixed directly into this local
quality comparison, because the order-only control exposes numeric sensitivity.
Initial-noise SHA is required within each matched group. All three methods get
the same generic packing/cache system and explicit4GiB cache ceiling (actual
occupancy differs); Dense has history density1, the two sparse methods0.25.
No captures are enabled in this suite. New per-method density mapping preserves
existing case-level overrides and permits one model load per GPU lane.

No formal holdout, Pareto admission or 477/957 quality claim follows from this
candidate's name or its capture result. Full-video and longer-horizon evidence
remain required. Treat candidate-order-only divergence as a numerical control,
not as a semantic improvement or an admission change.
