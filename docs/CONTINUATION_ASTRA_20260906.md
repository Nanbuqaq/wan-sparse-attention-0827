# Continuation checkpoint: LongLive system research

The original full user plan remains in scope and **is not complete**. Do not
repeat old 44/102-case matrices or overwrite pre-fix experiments. No training.
Both local GPUs and official InferHub submission remain authorized. Use physical
GPU locks and distinct cases; do not spawn agents without explicit authorization.

## Read first

1. `docs/D2H_READINESS_ERRATUM_20260906.md`: confirmed future-state correctness bug.
2. `docs/REAL_SYSTEM_RESULT_20260905.md`: measured results and evidence limits.
3. `docs/TETHERMEM_BOUNDARY_20260905.md`: source mask-time discrepancy and probes.
4. Outer `docs/EXECUTION_REGISTRY.md` and the weekly shared status board.

## Sources and current gates

- Original baseline remains `59020c0610cb48ffdf0ce36c32aba97378aa0b72`.
- `995b57a`: fixes legacy D2H readiness before CPU V means. Large forced-eviction
  gate now checks raw KV, future prototypes, output, route, and real cache updates.
- `98c26a1`: corrected eight-case calibration, actual online-context capture,
  initial-noise SHA checks, common bounded pageable archive. **8/8 completed**.
- `eade645`: bounded 477-frame validation, two methods. **2/2 completed**.
- `b66f03d`: dedicated-compute-stream overlap diagnostic.
- `2688bc0`: actual Nsight overlap auditor and all-five static candidate replay.
  Verify live HEAD/status before starting work; later documentation commits may
  follow these source commits.

## Verified results

- Original paired 477-frame tests: Dense 848.414 -> 253.691 s; Final 268.629 ->
  217.165 s. Exact latent/video bytes and all 6000 ordered routes match.
- Bounded replacements: Dense 257.323 s; Final 217.581 s, identical latents and
  ordered routes to cached references. Pinned archive goes from about 31 GB to
  zero; live staging about 86/43 MB. CPU archive still about 31 GB per 477 case.
- H39 Dense ablation: 182.987 / 175.010 / 67.490 / 55.581 s for legacy / candidate
  gather / archive runs / plus cache. 4/4 and exact outputs. Runtime calls itself
  H800 but reports 139.8 GiB; do not relabel this as confirmed H200.
- Pre-fix calibration: 9 pass / 1 Top-p-state runtime fail / 0 missing. Retained
  intact. Top-p motion took 680.5 s; the batch hit `infer_gpu_idle` at 20 minutes.
- Corrected calibration: noise SHA matches all eight; captured V prototypes
  match committed V exactly. Final beats peak on both categories; count helps
  motion but regresses state. Neither is promoted.
- All-five static replay on repaired early/late captures is complete. No global
  winner. Inspect per-layer/history effects instead of claiming utility success.
- Same-route batched FA2 staging pilot is exact and promising for fragmented
  groups; cold Dense is negative. It is not integrated in video and is not KVOut.
- True D2H/FA2 overlap is verified only with a dedicated compute stream in the
  component pilot (2.203 ms). CPU commit remains critical; no video overlap claim.

## Important artifacts (outer results root)

- `videos/readiness_repair_98c26a1_h/`, `manifests/readiness_repair_98c26a1_states.json`
- `metrics/readiness_repair_quality_98c26a1/`
- `metrics/repaired_static_rescreen_2688bc0/`
- `videos/bounded_archive_state120_eade645/`
- `metrics/bounded_archive_{dense,final}_vs_cached.json`
- `metrics/late_history_pipeline_v2/`: full six-frame captures at latent30;
  a separate full-generator trace at latent36, with no capture inside that call.
- `metrics/offload_overlap_pilot/`: raw samples, traces, SQLite and overlap audits.
- `metrics/pooled_archive_gate/`: pre-fix mismatch and repaired gates.

## Remaining work, in useful order

1. Use corrected schema-3 captures for any further method inference. They include
   actual query summaries and K/V prototypes; old captures cannot retrospectively
   prove prototype readiness. The earlier phase-prototype pilot used pre-fix
   trajectories and is not a promoted method.
2. Decide whether the small deep/late utility benefits merit one explicitly
   frozen layer/history-conditioned candidate. Otherwise retain legacy Final.
   Do not retune using formal held-out results.
3. Finish reuse-axis characterization, verified prefetch and any worthwhile
   runtime overlap. Current flags still reject unimplemented overlap; the pilot
   alone does not authorize an end-to-end speed claim.
4. Attention kernels are only a few percent in the observed first-pass generator
   traces. Optimized KVOut/video expansion remains gated. A Python group-outer
   online-softmax reference is not a KV-stationary kernel; do not use it to claim
   the requested 72-point crossover study has been completed.
5. Tether: sources/checkpoint/dependencies are ready; official unit tests and two
   automatic prefix masks are tested. Full oracle, causal propagation/query-role
   accuracy, external VAE/SAM2 cost and three-memory deletion tests are unfinished.
6. Freeze final online configurations only after method decisions. Formal
   holdouts are already frozen, but formal 477/957 matrices have **not** started.
   At most 4 online configurations; no oracle in online Pareto. Give Dense the
   same generic safe system optimizations when making algorithm comparisons.
7. Complete formal quality/late-quarter review, profile/counters, figures, terminal
   audit and paper attribution. Do not call the full plan complete beforehand.

## Frozen holdouts

`configs/formal/system_holdout_prompts.json`, SHA
`6d898e96dd28e924622d5585ab6cc85b139560282c01761cde87b518f3d17fea`:
astronaut, glassblower, fox, blue canvas. 477 seeds 20260908/20260909; 957 seed
20260910. Candle is stress-only. Dense screening remains valid; it does not use
the raced CPU V-prototype reduction.

## Operational notes

Read `/kaimm-distill/infer_hub/SKILL.md` and README before new submissions.
Official path is `sudo -n -E python3 /kaimm-distill/infer_hub/lib/infer_submit`;
never run project code with sudo. Freeze/test/dry-run/push, then one batch per
homogeneous stage. Inspect queue live; the batches listed above are completed.
Shared `job_mem_limit_gb` was observed as 0, not the local 256-GiB cgroup limit.
Hardware labels can disagree with VRAM; preserve runtime facts.

Outer scripts preserve frozen worktrees and cases. Late-capture attempt v1
failed before generation on DictConfig JSON serialization; v2 passed after
`ee141ca`. Preserve both. Do not rebuild old case identities using new system
dataclass defaults: use their frozen manifests and actual source case keys.
