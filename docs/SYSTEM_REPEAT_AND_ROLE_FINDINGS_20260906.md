# Repeated system gain and boundaries for causal memory

The original LongLive research program remains unfinished. Formal sparse477/957
holdouts remain sealed. No training, old-result overwrite or new admission
promotion occurred. Current figures and their five SHA-locked inputs are in
`results/metrics/system_role_research_report_20260906/` (outside this Git root).

## Same-route system improvement survives repetition

Frozen generator sources: old `2ad4465`, shared-union compiler `21393f3`.
The latter pre-collapses groups ONLY when they already consume the same union;
it never replaces per-query Top-p with a global budget. Each prompt used one
physical4090 and eight GPU-local physical CPU cores, two independent processes
per source. Thread settings were unchanged at2; CPU affinity is in the launch
manifest. Model load remains separate from complete generation/VAE/artifact time.

| Prompt | Successful order | Original complete seconds | Compiled complete seconds | Median reduction |
|---|---|---|---|---:|
| motion | ABBA |206.936,208.458|195.553,195.129|5.949%|
| state | BBAA (recovery changed order) |205.354,205.736|193.020,192.926|6.116%|

All8 videos pass, every pair has identical latent/video bytes,6000 ordered
routes and14,161,305,600 H2D bytes. Three attempts failed the idle-GPU check
before model loading, because the preceding process had just exited; they are
preserved. A bounded idle wait was added to the scheduler, and only those three
unexecuted cases were recovered. There are0 missing successful executions.
State is explicitly NOT described as a completed BAAB timing experiment.

Routing medians fall43.293→30.933s and42.856→30.570s (~28.6%). CPU gather medians
fall7.419→7.019s and7.710→7.474s. The earlier single-pair motion gather regression
does not reproduce. Attention is ~0.07%/~0.04% higher and complete backend ~0.41%/
~0.20% higher; do not round these into an assertion that every component improved.
With n=2, no inferential confidence interval or universal speedup is claimed.
Service times are not added as if they were independent critical-path waits.
The old unbound pairs are not pooled with these affinity-controlled repetitions,
and faster model loading here is not attributed solely to affinity.

Audit: `results/metrics/shared_compiler_repeats_audit_20260906.json`.
This supports a lossless shared-union compilation contribution, NOT a new
utility/admission algorithm or complete Causal Reuse-Aware Paging claim.

## Short-video bootstrap does not generalize reliably

Long calibration15/15 technical pass,0 missing, same-device Dense/Final/bootstrap
triplets and equal actual sparse budgets. Only2/5 independent prompt-seed groups
pass full/late LPIPS and latent-L2 non-regression. Motion16/state17 on H200 and
state18 locally regress. Bootstrap remains negative for promotion; no957
expansion, no tuning using formal cases. Asymmetric cohorts are not pooled.
Audit: `results/metrics/bootstrap477_quality/decision.json`.

## Mask availability is distinct from useful role prediction

The automatic SAM2 state teacher propagates153/153 frames with no empty masks.
Frames8/76/152 visually follow the measuring jug and its handle; changing liquid
is inside the same mask. This is not a separate state label or semantic ground
truth. Motion has no eligible automatic central-foreground initializer. The
earlier missing-OmegaConf attempt remains a preserved infrastructure failure.

Physically prefix-only directories of9/69/117 frames produce exactly the same
prefix masks as the153-frame teacher. This supports this producer's prefix
invariance at the two actual capture frontiers, not arbitrary-video causality,
VAE prefix equivalence, or integrated online generation. Prefix reprocessing
cost grows with length; it is not an efficient incremental implementation.

The fixed role formula was tested on40 complete captured calls on the EXACT
Dense actor that produced the masks. No sparse-trajectory masks were silently
substituted. Current mask labels enter only after predictions; historical masks
are still teacher-assisted, and CPU reductions are diagnostic rather than
bitwise GPU-proxy reproduction. Query groups can straddle latent boundaries;
the last8-query group is weighted by its actual token count.

| Predictor | Soft-mask MAE | Balanced accuracy |
|---|---:|---:|
| previous completed spatial mask |.00209|.99392|
| previous area constant |.15802|.50000|
| raw Q + history prototype |.14915|.60278|
| RoPE Q + history prototype |.14376|.75512|
| 50% raw Q + spatial prior |.07439|.99240|
| 50% RoPE Q + spatial prior |.07170|.98434|

Q has some discrimination, but the uncalibrated cosine-sigmoid probabilities
are not more useful than spatial persistence in this fixed-camera case. This
does not prove Q lacks semantic information on moving scenes. No temperature,
layer gate or mixture was tuned to rescue this one prompt. The current formula
does not enter online video expansion. A future moving-subject test first needs
a reliable automatically initialized mask and independent calibration.

The dormant identity-first three-role primitive assigns state only outside
identity. The jug example shows why this assumption needs reconsideration:
identity and irreversible state can coexist in one region. No three-role
runtime should be promoted before equal-byte semantic deletion tests on two
state calibration cases distinguish that hypothesis from ordinary background.

## Previous-layer prediction is costly; temporal retention is more selective

The old prefetch analyzer incorrectly treated sparsely sampled0→9 layers as an
adjacent edge. It now accepts only true `l→l+1` pairs; historical raw outputs
remain preserved. New replay uses0→1→2→3 and first denoising calls only, never
counts forced per-chunk cache hits as natural prediction accuracy.

Predictions use only source coordinates and the current known coarse candidate
set. A25% physical candidate-token cap is applied per head BEFORE inspecting
the target route. Block64's final24-token tail has its real12,288-byte cost,
not32,768 bytes. All KV belongs to the target layer; no cross-layer KV sharing.

| Predictor | motion recall / precision | state recall / precision | total bytes / no-prefetch Block64 |
|---|---|---|---|
| previous adjacent layer |35.36% /37.15%|45.85% /48.05%|1.598x /1.496x|
| same layer, previous chunk, candidate-filtered |24.85% /73.11%|35.93% /72.92%|1.091x /1.133x|

The random baseline is restricted to the same target calls but can spend more
bytes than the filtered previous-chunk predictor. Do not call that an equal-
realized-budget comparison. These are traffic/prediction measurements, not
timeliness, exposed wait or overlap. Earlier route similarity did not establish
these costs. No prefetch video backend is promoted from this table.

Separate per-layer LRU raw-residency replay excludes the five-call reuse axis.
At32MiB, L0 first-pass byte hits are48.7% motion /63.1% state;64MiB gives58.3%/
70.7%. L19 has45.4%/59.3% and56.7%/68.9%. Six sampled layers do not justify an
all30-layer hit-rate estimate. Cache storage excludes assembled union and other
transient buffers; CPU archive remains unbounded. No saved-time extrapolation
uses the rejected cost model.

This motivates testing a bounded raw cache combined with the already-effective
per-chunk union, not replacing five-call union hits with costly reconstruction.
The existing raw cache still restores tokens individually; its complete GPU
cost must pass before any video use. Preparation/restoration/full-wall timing
has been added, and a real two-prompt cold/warm materialization gate is next.

## Next bounded work

1. Complete raw-cache full-cost GPU gate; preserve negative prototype results.
2. If restoration dominates, test batched/slab composition before video. Count
   backing allocations and evicted views, not only nominal logical cache bytes.
3. Causal semantic memory needs mask availability, VAE frontier equivalence,
   incremental refresh cost and equal-byte interventions as separate gates.
4. Retain all original remaining work: actual verified overlap, conditional
   QOut/KVOut, fair baseline optimizations, bounded total history, and eventual
   formal477/957 only for configurations that pass the relevant gates.
