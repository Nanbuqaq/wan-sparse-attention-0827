# Matched-trajectory diagnosis and bootstrap ablation

User instruction: continue autonomously; do not end a turn at an ordinary batch
boundary. The original full LongLive plan remains open and formal holdouts sealed.

## Verified sequence

1. `be00491` instrumented the original `cf5c25f` generator, method order and
   physical GPU assignments. All6 diagnostic repeats reproduce the original
   initial noise, **all latent bytes and ordered routes**.240 captures complete:
   layers0/9/19/29, latent18/30, four denoising calls plus clean commit. These
   are instrumentation repeats, not six new independent quality samples.
2. `eff339a` freezes Raw/Aligned routes on each captured actor's original GPU.
   `f7d81b4` replays48 sequences/240 calls on the frozen routes and includes
   Dense, Final and aligned actor trajectories. Route construction is outside
   teacher evaluation; both candidates use exactly the same per-head budget.
3. State L9/latent18 (one candidate frame) shows aligned/raw error ratios
   averaged1.0242(Dense trajectory),1.0095(Final),1.0096(aligned). All five calls
   regress in each case. Motion does not have this pattern. This local weakness
   alone does not establish the cause of final video quality.
4. An H2009-case replication at `f7d81b4` completed9/9, missing0. Each prompt/seed
   has same-card Dense/Final/aligned controls. Actual runtime reports H200,
   SM9.0,139.831GiB. Results show the original aligned benefit is not robust:

| Case | Final LPIPS | Aligned LPIPS | Final latent L2 | Aligned latent L2 |
|---|---:|---:|---:|---:|
| motion seed20260911 |0.09440|0.09425|0.34395|0.34390|
| state seed20260911 |0.04360|0.04282|0.17303|0.17418|
| state seed20260912 |0.05738|0.05782|0.20479|0.20355|

This asymmetric exploration has no pooled cross-category mean; it does not
discard the failing original seed or establish a general benefit.

## Causal bootstrap intervention, source `2ad4465`

Two predeclared policies use Raw only when coarse candidate count is exactly1:
all layers, or only layer9. Later/multi-candidate calls remain aligned. Raw backup
prototypes are generated causally at archive insertion, with extra CPU metadata
bytes/index time counted. No selector reconstructs them from current full KV.

Four GPU branch gates and four actual153-frame videos pass. Causal checks pass:
the all-layer intervention matches the Raw control's latents through the first
sparse chunk; the L9-only intervention's first changed Q/exact KV/history KV
match the aligned control exactly, and it executes the frozen Raw coordinates.

| Case | Policy | LPIPS | Late-quarter LPIPS | Latent relative L2 |
|---|---|---:|---:|---:|
| motion seed20260904 | Final control |0.09156|0.25874|0.29827|
| motion seed20260904 | aligned control |0.07563|0.21267|0.26704|
| motion seed20260904 | Raw all layers at startup |0.07509|0.20968|0.26532|
| motion seed20260904 | Raw L9 only at startup |0.07383|0.20758|0.26338|
| state seed20260904 | Final control |0.05843|0.13084|0.23054|
| state seed20260904 | aligned control |0.06246|0.15029|0.24588|
| state seed20260904 | Raw all layers at startup |0.05636|0.12598|0.22501|
| state seed20260904 | Raw L9 only at startup |0.06168|0.14773|0.25152|

Only the all-layer startup policy passes the original two-category development
gate. L9-only does not explain/repair the state regression, despite its local
teacher advantage. This supports further startup-stage investigation, not a
universal layer9 rule or formal promotion. Absolute semantic quality and longer
independent trajectories still require validation.

## Lossless system finding, source `21393f3`

Real CPU route profiling shows58% of profiled route time in `build_route_plan`,
largely compacting888 duplicate query-group rows into12. For the explicitly
shared-union case only, the producer now emits one row per head and retains
virtual before/after counts. Per-query choices are NOT replaced by a global
budget; this applies only where all groups already consume the identical union.

All tensors, metadata and SHA match32 real frozen plans. CPU route median on
the tested state L9 six-frame capture:24.566→11.766ms (2.088x). Small/large real
GPU gates pass. This is not yet an end-to-end speedup claim.

## Completed long validation (do not resubmit)

Both original local system pairs completed4/4. Full latent/video bytes and all
6000 ordered routes match within each pair. Motion complete time216.853→210.036s
(3.144% lower); state222.718→204.905s (7.998% lower), unchanged H2D payload
14,161,305,600 bytes. These are single pairs, not a general speedup estimate.
Motion gather increased11.772→13.183s; the no-other-component-regression gate
is therefore unresolved. A distinct affinity-controlled ABBA/BAAB repeat batch
is running under `results/videos/shared_compiler_affinity_repeats_20260906/`.
Frozen manifest SHA `d3fc2df02da749ed65024ef1f7589988a10f317ceb1f43459ae4421578069131`.

Bootstrap long calibration completed15/15 technical pass, missing0: H2009 cases
(seeds20260916/17) and local4090 six cases (seed20260918). Same-card triplets,
common noise, equal actual sparse budget and metric/video hashes are checked in
`results/metrics/bootstrap477_quality/decision.json`. Only2/5 prompt-seed groups
pass full/late LPIPS and latent-L2 non-regression. H motion16 and state17,
local state18 regress. **Bootstrap is not promoted and does not expand to957.**
No pooled cross-hardware latency or asymmetric cross-category quality mean.

Historical launch records:

- Local source pair `2ad4465` vs `21393f3`, Final120 latent/477 pixel, same GPU
  per prompt. GPU0 motion old→new; GPU1 state new→old. Output root:
  `results/videos/shared_compiler_21393f3/`; launcher
  `scripts/run_shared_compiler_video_pair.sh`. Completed, no resubmission.
- H-pool bootstrap477 calibration9 cases, source `21393f3`, submitted once after
  dry-run: `zhouhe08__longlive_bootstrap_calibration477_Iter0__21393f3d0ad1`.
  Fresh seeds20260916/20260917, Dense/Final/all-layer-bootstrap on same-card
  triplets. Shared output:
  `/kaimm-distill/zhouhe08/longlive-system/outputs/bootstrap-calibration477-21393f3-h-v1`.
  Not formal holdouts. Frozen publisher entry supports the three3-method lanes.

## Evidence roots

- `results/metrics/matched_trajectory_capture_be00491/trajectory_audit.json`
- `results/metrics/cross_route_plans_eff339a/` and `cross_trajectory_replay_f7d81b4_local/`
- `results/videos/aligned_seed_replication_f7d81b4_h/recovered_audit.json`
- `results/metrics/aligned_seed_quality_f7d81b4/`
- `results/videos/bootstrap_ablation_2ad4465_local/{terminal_audit,causal_audit}.json`
- `results/metrics/bootstrap_quality_2ad4465/`
- `results/metrics/route_cpu_profile_2ad4465/complete32_equivalence.json`
- `results/metrics/compiler_gpu_gates_21393f3/`

Input staging note: the user's shared input parent was not writable by the
current account. No chmod/sudo project-code workaround was used. Fixed-capture
replay stayed local using read links; InferHub ran new inference using its
already-shared read-only model bundle. All raw results remain preserved.

## SAM2 teacher boundary

`f9d5ef0` automatic state teacher propagated153/153 frames without empty masks.
Assistant review of frames8/76/152 finds the mask follows the measuring jug,
including its changing contents; it does not separately label liquid state.
Motion initialization is negative (no eligible central foreground). Earlier
missing-OmegaConf launch failure is preserved, not counted as a mask result.
Full results: `results/metrics/sam2_oracle_f9d5ef0_v2/`. No Tether video inference
or online role-quality/speed conclusion follows from this producer gate alone.
