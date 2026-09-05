# First complete-video system result and executor follow-up

## Result, not yet a general paper claim

One isolated development case, `calibration_state`, seed 20260904, 120 latent /
477 pixel frames, RAG Dense, BF16 grouped FA2, source `acff738e0118f84fb5a96f6753e7b3f05d5be33b`.
The two configurations ran on distinct local RTX 4090 GPUs with two CPU threads
per process, from one detached checkout. No formal holdout or sparse routing was
used. Both cases technically pass; no fallback, failed call or missing case.

| Measure | Legacy materialization | Archive runs + per-chunk roped cache |
|---|---:|---:|
| Generation, VAE, artifact write/encode/decode audit | 848.414 s | 253.691 s |
| Including this run's model load | 1143.568 s | 551.218 s |
| Peak allocated device memory | 11.908 GiB | 14.318 GiB |
| Counted historical KV H2D payload | 283.226 GB | 56.645 GB |
| Cache hits / misses | not enabled | 4080 / 1020 |

Warm-runtime speedup is **3.344x**, reduction **70.10%**. This particular cold
start is **2.075x**, with roughly 295–298 s model loading reported separately.
These are one-pair measurements, not confidence intervals or cross-hardware
estimates. The lanes share host resources and have fixed GPU assignments; a
controlled repeat/crossover is still needed for a publication-quality estimate.

Correctness evidence is stronger than a perceptual threshold: all latent bytes
and both video files are identical, and the **ordered** 6000-call route sequence
matches (not just its unordered set). Latent max-abs, relative-L2 and cosine
distance are zero. Cache occupies 2,668,723,200 bytes across 30 entries, with no
eviction under its 4 GiB budget. H2D payload is accounting, not Ncu HBM traffic.

Facts: `results/videos/dense_state120_acff738/{states,terminal_audit}.json` and
`results/metrics/astra_audit/dense_state120_acff738_comparison_v2.json`.
The preceding comparison JSON is preserved; its FP32 cosine reduction produced
a small invalid negative distance for identical huge tensors. Float64 reductions
fix that audit metric without changing generated results or speed measurements.

## Attribution and measurement limits

- This is **generic system optimization of Dense**, not a new sparse algorithm.
  Admission is unchanged; runs, persistent staging and cache form one combined
  intervention. Their independent contributions are not yet isolated here.
- Existing four-lane 39-latent Dense ablation remains queued at `a473e34`; it
  will separate candidate gather, archive packing, and cache without re-running
  the reserved Dense motion case in the ten-case routing calibration.
- Legacy `materialize_total_s=0` means that complete subcounter was not populated,
  not zero materialization cost. Candidate concatenation in the runtime wrapper
  also lacked a named counter; a subsequent instrumentation-only patch adds
  `candidate_prepare_s`. The top-level measured wall time already includes it.
  Do not add partial/nested service counters to infer a critical path.
- A same-admission Final system comparison is the next baseline fairness check.
  New admission and system optimization still need the planned 2x2 attribution;
  this result alone cannot establish the full causal paging algorithm claim.

## Visual evidence boundary

The identical video was fully decoded; overview and all four quarters were
reviewed at 16 frames per quarter (assistant visual review, not human labels).
No cut, long freeze or flicker event was flagged by the simple frame diagnostics.
Cup identity and committed filled state persist. However, it fills in the first
quarter, then the inflow continues with a nearly plateaued level; the red liquid
also darkens substantially. Thus relative Dense fidelity is exact, but this is
weak evidence for *continued* long-horizon state updates. It must not by itself
validate the identity/scene/state lifecycle hypothesis. No formal prompt or
holdout SHA was changed after observing this development video.

## Why the next executor experiment is staging, not KVOut

A bounded five-call Final Top-p 0.95 Nsight trace at `4b6f976` shows 1.074 s in
the complete grouped backend, of which 0.913 s is per-group KV preparation;
FlashAttention kernels themselves total 0.0174 s. There are 9693 stream sync API
calls. Nested NVTX scope percentages are not additive. This is a synthetic
runtime trace and includes profiler overhead, not a full-video breakdown.

The follow-up `dd03600` replaces the Python/CUDA per-group pack loop with batched
index gathers and one output scatter. The FA2 kernel, route, Q/K/V and output are
unchanged. Six CPU geometry/striding tests and a small CUDA gate passed before
four large resident replays (Q=4680, exact=9360, history=9360, H=12, D=128;
5 warmups, 30 interleaved measurements):

| Regime | Legacy median | Batched cold recipe | Reused CPU recipe |
|---|---:|---:|---:|
| Dense | 10.162 ms | 12.302 ms | 9.359 ms |
| Shared union | 6.988 ms | 7.500 ms | 4.809 ms |
| Fragmented groups | 189.834 ms | 98.863 ms | 25.621 ms |
| Strided inputs | 190.606 ms | 95.672 ms | 25.944 ms |

All outputs are bitwise identical. Fragmented peak allocation falls from 8.954
to 4.625 GB. **Cold Dense/shared recipes are negative**, so do not enable this
globally. The repeated CPU recipe still transfers metadata each call, and its
first construction must be charged once per reuse lifetime. It is not a
KV-stationary implementation, does not eliminate final per-group KV replication,
has not been integrated in video runtime, and supplies no HBM-counter claim.
Large fragmented metadata is about 69 MB: compact descriptor/GPU index generation
is a possible next experiment, not a hidden free cache or measured benefit.
