# First complete-video system result and executor follow-up

**Later correctness erratum:** [D2H readiness](D2H_READINESS_ERRATUM_20260906.md)
found a race in legacy CPU V-prototype construction. The pre-fix calibration
rankings below are provisional, not final method-selection evidence. Recorded
exact-output system pairs and all raw artifacts are preserved with their code
versions and limitations.

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

## Completed follow-up: Final and four-lane attribution

The Final pair at `930f663` also passed exact latent/video and ordered-route
equality on the same 477-frame state development case: **268.629 -> 217.165 s**,
1.237x / **19.16% less complete time**. Cache hit/miss counts remain 4080/1020;
live cache payload is 667,180,800 bytes under the configured 768 MiB cap.
Peak allocated memory changes from 11.908 to 12.511 GiB. See
`results/videos/final_state120_930f663/comparison.json`.

The four-lane `a473e34` H-pool 39-latent Dense batch has now completed 4/4 and
was recovered into `results/videos/dense_system_validation_a473e34_h/`.
All three interventions preserve every latent/video byte and ordered route:

| Dense configuration | Complete time | Historical KV bytes |
|---|---:|---:|
| Legacy | 182.987 s | 50.319 GB |
| TransferPlan + candidate gather | 175.010 s | 50.319 GB |
| Archive-run packing | 67.490 s | 50.319 GB |
| Archive-run packing + cache | 55.581 s | 10.064 GB |

Packing is the largest improvement here; cache independently reduces time a
further **17.64%** relative to packing-only. The small 4.36% TransferPlan-only
change is below the 10% end-to-end promotion threshold. These are concurrent,
single-case lane measurements, not repeated trials. Runtime reports **H800**,
compute capability 9.0, but about **139.8 GiB** VRAM. Preserve this inconsistent
label exactly; do not silently call this confirmed H200 evidence or mix its
absolute latency with the 4090 measurements.

## Calibration terminal outcome and decision

The `4b6f976` batch was killed by the platform at 20 minutes (`infer_gpu_idle`,
reported mean GPU utilization 2.2%). Eight new cases completed; Top-p state did
not. Individual artifacts were recovered rather than depending on the killed
batch's missing final merge. Including the explicitly reserved `a473e34` Dense
motion reference, the ten-case protocol is **9 pass / 1 runtime fail / 0 missing
terminal states**. No successful case was regenerated.

Full-video LPIPS / latent relative L2 versus Dense:

| Method | Motion LPIPS / latent L2 | State LPIPS / latent L2 |
|---|---:|---:|
| Legacy Final | 0.08367 / 0.28374 | 0.05702 / 0.20757 |
| Top-p 0.95 | 0.11021 / 0.33106 | runtime incomplete |
| Peak utility | 0.09009 / 0.29822 | 0.06017 / 0.22385 |
| Count-uniform utility | 0.07268 / 0.25920 | 0.05978 / 0.21724 |

Neither utility satisfies the two-category non-regression gate. Count's motion
improvement is retained; it is not discarded merely because state regresses.
Top-p's completed motion case takes 680.5 s (legacy's 66.0 s is itself capture-
augmented, so not a clean speed baseline). It is not a formal candidate.
LPIPS uses the locked 0.1.4 package, AlexNet/linear SHA-verified weights and exact
Torch/torchvision version checks; old package/weight locations were read-only.

Complete attention replay now includes exact/current/recent KV, actual-byte
legacy controls and per-query retained mass. **All eight new captures are at
latent start 18 with only one historical candidate frame** (1560 history, 9360
exact tokens), not six-frame steady state. Utility uses 344 tokens/head at this
point versus legacy's 390: whole-block/tier quantization is more substantial
than the roughly 24.63% video-average budget suggests. Matched-byte controls
rerun the actual legacy selector, not a coordinate-prefix truncation. Further
late-history captures are required before freezing any new admission.

## Independently motivated phase-prototype pilot

For `upstream_zero`, historical temporal RoPE is always zero and spatial indices
are immutable. Thus index-time rotated K means are causally computable without
onloading complete candidates during selection. The pilot reconstructs every
executed K exactly from raw archived K and canonical frequencies before using
these prototypes with current post-RoPE Q summaries.

Same-byte RoPE-aligned Final improves layer-0 relative L2 by 6.16% (motion) and
3.35% (state), with higher p05 retained mass. Extending to four layers reveals
mixed results: all motion layers improve, while state layers 9/19/29 regress by
2.40% / 0.29% / 1.79%. This is **not promoted**, is not integrated online, and is
not evidence that all layers should use one prototype representation. The raw-
K reconstruction and fixed-policy invariance tests remain useful results.

## Bounded-memory and overlap follow-through

The repaired `eade645` 477-frame Dense/Final runs preserve all latents and
ordered routes against their earlier cached counterparts. Complete times are
**257.323 s / 217.581 s**; the differences from the earlier cached runs are
1.43% / 0.19% slower in these single pairs, not an additional speed claim.
Each retains 31,054,233,600 bytes of CPU archive, but **zero pinned archive
payload**. Live persistent staging is **86,261,760 / 43,130,880 bytes**. Total
CPU archive memory still grows with duration; bounded pinned memory does not
establish bounded total streaming memory.

Two-slot D2H plus pageable-commit replay uses the same real captured route and
resident FA2 inputs. All transfer data and Attention outputs match exactly.
The default-compute-stream path reduces wall time from 15.88 to 9.80 ms but
Nsight measures **zero** GPU copy/Attention intersection. A dedicated compute
stream produces **2.203 ms** of actual overlap, with unchanged 57,507,840 D2H
bytes (2.563 ms copy service, 4.841 ms Attention service in the traced trial).
Changing maximum CUDA connections to 32 did not produce overlap in the default
path. The driver/default-stream cause is not fully resolved.

The dedicated-stream unprofiled median is 10.00 ms: true GPU overlap does not
materially beat the already scheduled CPU-commit path because CPU commit remains
critical. This is a **component pilot**, excluding prototype indexing, model and
VAE, not integrated video overlap and not KVOut or H2D evidence. Facts:
`results/metrics/offload_overlap_pilot/{default_stream_audit,dedicated_stream_audit}.json`.

After the readiness fix, all five preregistered static utility candidates were
re-screened on both categories, two layers and early/late history. No uniform
candidate dominates legacy across these points. Some deep late-history points
improve, motivating a possible layer/history-conditioned hypothesis, but no such
method is implemented or promoted. Facts:
`results/metrics/repaired_static_rescreen_2688bc0/`.
