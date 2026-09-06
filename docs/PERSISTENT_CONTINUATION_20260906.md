# Persistent full-plan execution: current checkpoint

User explicitly requires execution until the FULL original plan completes or
they manually stop it. Do not return a final merely at another small milestone.
Both local GPU0/GPU1 and official InferHub are authorized. No subagents were
authorized. Preserve old experiments/read-only checkouts; no training.

## Current source and priority

Public branch `longlive-system`, latest source at this checkpoint `59540bd`.
CPU regression311 passed/1 skipped. Original goal is NOT complete. Formal sparse
477/957 matrices are still unstarted; holdout SHA remains
`6d898e96dd28e924622d5585ab6cc85b139560282c01761cde87b518f3d17fea`.

1. Resolve preencode RGB differences in a frozen H diagnostic already submitted.
2. Finish oracle video visual/quality interpretation and causal stop gates.
3. Finish genuine QOut/KVOut reference/conditional characterization and system
   gate decisions; never label the old Python group-outer reference KV-stationary.
4. Freeze eligible online methods (at present, no new admission/role/hierarchy
   promoted) and execute/audit formal477/957. Keep generic safe optimizations
   available to Dense. Oracle never enters online Pareto.
5. Finish attribution, timelines, plots, semantic late-quarter review, terminal
   audits and paper handoff. Full plan only complete after these deliverables.

## Raw cache implementation and real results

- `d05fa6c`: vectorized request compilation + batched restore, independent
  owning cache entries. Warm~5.9ms rather than639/653ms, cold22–26ms.
- `3c437d9`: bounded token-valid GPU slab; copies selected missing tokens only,
  not full Block64 overfetch. Exact KV padding0; indices charged separately.
- `0d4d109`: vectorized missing-bit/run construction, named tuple storage keys.
  Real4090 cold~10ms, warm~3.85ms; uncached archive runs8.6–8.8ms.
- `8d2f99a`: hierarchical per-chunk roped union above raw slab. Raw budget is
  within the total explicit GPU cache budget. `SlabTransferPlan` is a physical
  plan, not a selector or cost model. All backing allocations are counted.
- Small/large Final and large Dense real two-chunk gates pass. Each includes
  five calls/chunk, new archive evictions and original K/V/prototype equality.
  Union reuse8hit/2miss across two chunks; cross-chunk raw hit branch reached.

## Completed H200 eight-video hierarchy batch: mixed, NOT promoted

Source`93db4ab1c7a85b1af730bd14bcdd8d1c7e7d3e17`, job
`zhouhe08__longlive_hierarchical_pairs_Iter0__93db4ab1c7a8`, completed8/8, missing0.
Each same-card loaded pair: Dense/Final x motion/state,120 latent/477 pixel,
seed20260904. Total configured cache4GiB; hierarchy raw1GiB + union3GiB.

| Case | Union-only seconds | Hierarchy seconds | KV+index onload reduction |
|---|---:|---:|---:|
| motion Dense |214.113|244.059| -4.22%|
| motion Final |159.418|160.853|33.48%|
| state Dense |236.154|237.973| -4.22%|
| state Final |160.906|155.813|41.89%|

All4 pairs have EXACT same initial noise, all latent values and ordered routes.
But MP4 hashes AND decoded RGB differ (u8 RMSE1.47–2.17). Do not falsely say
all video bytes match. Do not blame KV: latent/route equality is proven.
Original runs lack preencode RGB, so attribution to VAE vs encoding is OPEN.
The audit originally raised on MP4 mismatch; updated audit distinguishes layers.

Facts recovered under `results/videos/hierarchical_pairs_93db4ab_h/`:
`recovered_states.json`, `recovered_audit.json`, complete original files/logs.
Metrics: `hierarchical_pairs_93db4ab_audit.json`,
`hierarchical_video_payload_audit_93db4ab.json`.

Dense slab barely hits:1GiB is below its complete30-layer raw working-set sweep;
per-layer trace hit rates cannot be extrapolated into that undersized global LRU.
Even Final's substantial onload reduction does not give robust complete-time
improvement. Hierarchy stays mixed/negative, not a formal configuration.

Encoder-only repeats using the EXACT shared PyAV15/libx264 core165 at both81
and477 frames produce identical files/pixels on each of three identical-RGB
inputs (default and single-thread modes separately). This does NOT prove the
original encoder was uninvolved, but rules out casually asserting nondeterminism
without reproducing it. A noncontiguous PyAV frame roundtrip also passed.

## ACTIVE raw-RGB H diagnosis — do not resubmit

Frozen source `59540bd0b8c98ac71f4cc5f1ab19e273c39dfa0a`.
Job `zhouhe08__longlive_raw_rgb_diagnostic_Iter0__59540bd0b8c9`.
2 assigned H GPUs,4 cases: same motion Dense/Final system pairs. Same latent
length120/seed20260904. These are INSTRUMENTED REPEATS, not quality samples or
new speed measurements. System profile_mode=trace differentiates case identity.

Shared root:
`/kaimm-distill/zhouhe08/longlive-system/outputs/raw-rgb-diagnostic-59540bd-h-v1`.
Local submission records:
`results/infrastructure/inferhub/raw-rgb-diagnostic-59540bd/`.
It records float-video and preencode-u8 RGB hashes, RGB strides, and saves full
`raw_rgb_frames.pt` (~571MB/case). Compare these BEFORE attributing pixel changes.
Raw hash is also now recorded for future ordinary runs (dump only when requested).

## Oracle Tether is now actual runtime, not just mask production

- `3f545ea`: offline method ID, compact two-query-subset SDPA backend, separate
  source-compatible mask ADDRESSING vs latent-aligned teacher, matched-Dense
  hash guard, case-specific method filtering with correct load amortization.
- GPU v1 failed from bias/query dtype mismatch before any video. Failures/logs
  preserved in `results/metrics/oracle_runtime_v1/`.
- `b20d0e1`: dtype fixed to match pinned CUDA SDPA/public source. Both actual
  large two-chunk Oracle gates pass with independent FP32 biased Attention;
  `results/metrics/oracle_runtime_v2/{source,aligned}.json`.
- Frozen local snapshot `/tmp/longlive-oracle-b20d0e1`, outer launcher
  `scripts/run_oracle_tether_videos.sh`. Three videos COMPLETE3/3, missing0:
  new Dense matches original reference video SHA EXACTLY, then source-addressing
  oracle and latent-aligned oracle. Each39 latent/153 pixel, state seed20260904,
  all on physicalGPU1. No manual ROI; full history KV first transferred.
- Results root `results/videos/oracle_tether_b20d0e1_local/`;
  states `lane1/shard_0_states.json`, `terminal_audit.json`.
- Quality `results/metrics/oracle_tether_quality_b20d0e1/state.json` COMPLETE:
  source oracle LPIPS .053468 / late .124141 / latentL2 .227458;
  aligned oracle .055438 / late .130942 / latentL2 .233039.
  Not equal-byte comparisons with Final; not an online speed/quality claim.
- Three overview + twelve quarter storyboards COMPLETE under
  `results/metrics/oracle_tether_storyboards_b20d0e1/`; assistant inspection
  remains to be done. Do not omit target-average saturation: jug area>.25 makes
  source context-only target unattainable, forcing context weight0.

## Causal feasibility gates and limits

`3a09d89` VAE completed-prefix/incremental gate passes on BOTH prompts:3/18/30
latent prefixes and continuous3-latent decoding reproduce39-latent full decode
raw pixels exactly. Full decode~9s; incremental~8.6–8.7s. This is on recorded
latents, not yet integrated online generation. Facts:`metrics/vae_prefix_v1/`.
SAM2 state prefix-only9/69/117 mask invariance also passes; motion automatic
central initializer fails. The fixed Q-role formula loses to spatial persistence
on40 Dense-actor state captures. No causal subject runtime promoted; no claim
that Q universally lacks semantic information. Identity-first exclusive state
roles remain unvalidated; do not hide state-in-subject overlap.

`AttentionBiasPlan.digest()` now includes the actually executed context weight;
diagnostic timing metadata does not change its semantic digest.

## Operational reminders

- Both local GPUs were free after Oracle generation/gates; GPU0 quality scoring
  also completed. Recheck live processes before reuse. Only the H raw-RGB job
  above is intended to remain active at this checkpoint.
- InferHub skill, README and watchdog template were read IN FULL this turn.
  Use official `sudo -n -E python3 /kaimm-distill/infer_hub/lib/infer_submit`,
  tested/pushed SHA, dry-run then one frozen batch; no shared-env or worker edits.
- `rsync` is absent. Artifact recovery succeeded with
  `cp -a --no-preserve=ownership` into a new results directory, then
  `recover_system_case_states.py` path relocation and `audit_case_states.py`.
- `audit_case_states.py` prints huge JSON: redirect its stdout to a log rather
  than dumping the whole report into context. Read summaries selectively.
- Memory registry35–37 was consulted for IO/evidence boundaries; do not edit
  memory. No goal was created/marked complete.
