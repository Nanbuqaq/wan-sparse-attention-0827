# LongLive Sparse Attention

Current system-stage evidence: [complete-video system results and research boundaries](docs/REAL_SYSTEM_RESULT_20260905.md).
The matched 477-frame Dense and Final system pairs preserve all latent/video bytes
and ordered routes; observed complete-time reductions are 70.10% and 19.16%
(single development pairs, not general speedup claims). The ten-case routing
calibration has 9 passes and 1 explicit runtime failure; no new admission is
promoted. Earlier [preliminary profiles](docs/SYSTEM_PROFILE_PRELIMINARY_20260905.md)
remain historical evidence, with later accounting corrections in the new report.
Tether information boundary: [oracle versus causal routing](docs/TETHERMEM_BOUNDARY_20260905.md).
Latest correctness gate: [legacy D2H readiness erratum](docs/D2H_READINESS_ERRATUM_20260906.md).
V-aware calibration rankings are provisional pending the corrected batch.

Training-free sparse-history routing and rectangular attention backends for
LongLive and LongLive-RAG. The branch compares cache/transfer-aware
autoregressive variants of Block, five distinct clustering directions, SVG2,
AdaCluster, SVOO and SCOPE. The paper-oriented extension adds coverage-only,
online V-aware and transfer-bounded V-aware history routes.

The `longlive-system` branch extends that baseline with explicit online/offline
information boundaries, immutable logical routes, physical transfer plans,
versioned system-cost prediction and service-time versus critical-path traces.

## Evidence boundary

- `history_pair_density`, `history_transfer_density` and global executed
  density are separate metrics.
- Algorithms emit a backend-independent `HistoryRoutePlan`; kernel comparisons
  replay the same plan SHA.
- Native Dense has `routing_stage=N/A`; RAG Dense is post-transfer.
- Failed routing/kernel attempts remain explicit `fail` records and are not
  included in quality rankings.
- Online V-aware routing uses only compact Q summaries and CPU K/V prototypes
  before transfer. Exact output-residual scoring is an offline calibration
  teacher and is never read by the online route.
- Proposed-route CPU metadata is one K mean and one V mean per Block64; no
  token-level K-means index is constructed or tuned online.
- Q-summary granularity is selected only by isolated QKV calibration from
  64/128/256-token candidates before any formal sparse result is generated.
- `TransferPlan` may sort, coalesce, pad or cache a route, but it may not change
  the route's logical Attention edges or SHA.
- `SystemCostModel` predicts cost from a frozen hardware profile; measured
  exposed wait is post-run evidence and is never read by the current route.
- Dense-reference TetherMem masks are offline oracle evidence. Online methods
  may consume only completed-history masks, prototypes and causal summaries.

## Setup

The repository uses public pinned submodules:

```bash
git submodule update --init --recursive
```

Model weights are not included. For the InferHub input bundle, run:

```bash
LONGLIVE_CONFIG_PATH=configs/inferhub/rag_dense_21.yaml \
  bash scripts/inferhub_entry.sh
```

Batch entrypoints resolve `LONGLIVE_INPUT_BUNDLE_ROOT` explicitly (with
`INFER_WEIGHTS_DIR` accepted only as the platform-provided legacy alias) and
validate the overlay, model directory and both checkpoints before importing the
runtime. Public configs contain artifact ids and SHA-256 values, never
cluster-local paths.

CPU routing tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q -p no:cacheprovider tests
```

## Stage runners

- `scripts/inferhub_batch_gate_v3.sh`: isolated GPU routing and backend gates,
  Block64 100% and RAG cache-roll/QKV capture.
- `scripts/inferhub_batch_calibrate_methods.sh`: corrected RAG/Block replay,
  real-QKV paper calibration, all-method 21-frame smokes and warmed identical-
  route backend benchmarking.
- `scripts/inferhub_prepare_method_calibration.sh` and
  `scripts/inferhub_batch_complete_methods_v2.sh`: CPU-prep real-QKV
  calibration followed by four disjoint GPU lanes for matched 100% correctness,
  four paper-method smokes and warm backend benchmarking.
- `scripts/inferhub_batch_benchmark_routes_v2.sh`: four distinct real-shape
  density lanes, each replaying one immutable route across grouped, fixed and
  varlen backends with isolated compiler caches.
- `scripts/inferhub_batch_dense_screen.sh`: four disjoint GPU lanes for the
  Dense-only eight-prompt, two-seed Native Dense/RAG Dense screen.
- `scripts/build_dense_review_table.py` and `scripts/freeze_dense_prompts.py`:
  score each Dense video on five 0--2 criteria, take the worse Native/RAG total
  per prompt/seed, then freeze by two-seed mean, worst seed and prompt id without
  consuming any sparse result.
- `scripts/build_formal_suites.py` and `scripts/inferhub_batch_basic_477.sh`:
  generate and run two 477-frame base cases for 22 methods (44 cases) on four
  or eight disjoint lanes, including same-commit Native Dense and RAG Dense
  references. RAG Dense captures layers 0/9/19/29 at early/middle/late calls.
- `scripts/build_formal_basic_residual.py` and
  `scripts/inferhub_batch_basic_residual_8gpu.sh`: terminal-audit a completed
  checkpoint, subtract it from the original 44 identities, greedily balance
  only the remaining unique cases across eight GPUs, then merge old and new
  states without changing the frozen method-code commit. If one eight-GPU node
  fails the platform utilization gate because routing is CPU-heavy,
  `scripts/inferhub_batch_basic_residual_partition4.sh` runs two disjoint
  four-GPU partitions on separate machines and preserves the same task union.
  If the scheduler co-locates both partitions and recreates the CPU bottleneck,
  `scripts/inferhub_batch_basic_residual_full.sh` runs the complete residual
  plan as one four-GPU job, matching the resource shape that passed the gate.
- `scripts/local_formal_basic_residual_dual_gpu.sh`: reliable fallback for
  CPU-heavy routing that cannot satisfy InferHub's utilization gate. It runs
  two disjoint lanes on physically locked local GPU0/GPU1 without a global
  workflow lock, using a clean detached worktree at the frozen method commit.
- `scripts/build_video_review_storyboards.py`: fully decode every terminal
  477- or 957-frame video, verify its SHA/frame count, and emit an overview,
  four quarter storyboards and freeze/cut/flicker diagnostics for manual review.
- `scripts/audit_qkv_captures.py`: validate the frozen layer/start capture grid,
  Q/K/V shapes, BF16 dtype, finite values and per-file SHA before reuse.
- `scripts/recover_formal_basic_results.py`: after the 44-case terminal audit,
  non-destructively recover manifests, logs, videos, latents, QKV and review
  frames into the outer results tree, verify every copy SHA and re-audit the
  local video/latent artifacts with rewritten paths. An alternate expected
  manifest/count can reuse the same recovery gate for Pareto and final-long
  terminal batches.
- `scripts/calibrate_proposed_history_from_trace.py` and
  `scripts/build_proposed_long_calibration_suite.py`: rank initial 70/15/15
  and 80/10/10 candidates with isolated exact-output teachers, then build two
  non-formal 477-frame calibration cases before parameter freeze.
- `scripts/evaluate_videos.py`, `scripts/summarize_video_cases.py` and
  `scripts/analyze_complexity.py`: paired fidelity, complete-video bootstrap,
  a hard two-valid-base-case Pareto gate and separated theoretical/measured
  complexity. `scripts/build_case_metrics.py` joins technical, quality and
  manual evidence and preserves explicit negative outcomes. Formal LPIPS uses
  `configs/quality/lpips_alex_v0p1.json` and rejects unverified linear weights,
  AlexNet trunk weights or package versions before any metric is reported.
- `scripts/evaluate_formal_basic_quality.py`: group all reviewable base cases
  into matched Native-Dense and RAG-Dense comparisons, include each reference
  as its own metric row, and greedily split the four prompt/baseline groups
  across distinct local GPU ids without duplicating a case.
- `scripts/build_pareto_suites.py` and
  `scripts/inferhub_batch_pareto_expansion.sh`: freeze the five-density,
  four-prompt/two-seed, two-refresh/three-RoPE and four 957-frame expansions,
  with one same-commit RAG Dense mapping for every distinct prompt/seed/length.
- `scripts/build_pareto_partition_plan.py` and
  `scripts/inferhub_batch_pareto_partition8.sh`: split the 102 frozen cases
  into disjoint eight-GPU jobs, stagger GPU-active and CPU-heavy route starts,
  bound per-lane CPU threads, preserve exact-task terminal states, and fail the
  job if a runner omits a terminal state.
- `scripts/summarize_pareto_expansion.py`,
  `scripts/plot_pareto_expansion.py` and
  `scripts/audit_pareto_result_bundle.py`: build density/prompt/RoPE/refresh/
  long-video tables, PNG Pareto figures, the video index and the final
  cross-artifact audit.
- `scripts/inferhub_batch_pareto_route_benchmarks.sh`: replay selected methods
  on the frozen early/middle/late QKV snapshots with isolated cold-JIT caches
  followed by `5 warmup + 20 measured` iterations.
- `scripts/inferhub_batch_final_long_confirmation.sh`: run the second-seed
  240-latent/957-frame confirmation as four prompt-sharded lanes; every GPU
  executes one unique Native Dense, Native Block, RAG Dense, Block64 and final
  case, then all 20 outcomes are merged and terminal-audited.
- `scripts/audit_training_gate.py`: evaluates the frozen training gate against
  the supplied expected manifest. The completed Pareto run returns
  `do_not_train`; no post-hoc 50% density result is introduced to trigger
  MSE/LoRA.

Load-once runners write per-case terminal states and can reuse only successful
cases whose video SHA, full decoded-frame count and latent artifact all verify.
Formal case identities include the full code commit, prompt-content SHA, seed,
latent length, history density, RoPE policy, refresh policy and backend, so a
Dense reference can never be paired across commits. Local GPU commands use
`scripts/run_on_free_gpu.py`; independent lanes may explicitly select GPU0 and
GPU1 in parallel, while `/tmp/wan_sparse_gpu_<index>.lock` prevents two lanes
from claiming the same physical device. A global workflow lock is optional and
is not used by the parallel experiment lanes.
Native methods pair only with Native Dense; RAG/history methods pair only with
RAG Dense.

See `SOURCE_LOCK.json` and `THIRD_PARTY_NOTICES.md` for upstream commits,
licenses and modification boundaries.

The completed two-prompt 477-frame base-matrix status and mixed-hardware
evidence boundary are recorded in
[`docs/FORMAL_BASIC_477_AUDIT_20260902.md`](docs/FORMAL_BASIC_477_AUDIT_20260902.md).
The completed 102-case Pareto expansion, 957-frame results, negative outcomes
and training decision are recorded in
[`docs/PARETO_EXPANSION_AUDIT_20260903.md`](docs/PARETO_EXPANSION_AUDIT_20260903.md).
