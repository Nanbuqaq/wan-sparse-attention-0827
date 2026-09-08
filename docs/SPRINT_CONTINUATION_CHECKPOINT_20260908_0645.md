# Continuation checkpoint — 2026-09-08 ~06:45 UTC

Active goal remains through14:21UTC or user stop (about7h36m left). Do not mark
the sprint complete or stop background jobs. This turn made substantive progress:
new bounded KV-version experiment6/6, review closure, new957timing runs and native
LongLive2 setup. No subagents; no training; old44/102/formal results read-only.

## Live sources and work

- Main repo `/home/zhouhe08/MyProjects/0904-longlive-system/publish_repo`, branch
  `longlive-system`, latest local8308cfceb64ef3e16061219c1057ecb74bac9849. Verify
  live Git; last pushed3cd428991bea2c1a090163708b05b3b4f23c69ac. New commits need
  push before any newInferHub batch. Preserve untracked
  `scripts/audit_retrieval_divergence.py`; old0400/0500checkpoint docs are also
  untracked and should not be deleted/reset.
- GPU0 andGPU1 both4090 currently run `streaming957_local_v1`, respectively
  motion/state. Exec sessions96061/93418. At06:41 each lane completed4/6cases;
  remaining repetitions continue. Do not relaunch or change the runtime.
- Locked source worktree `/tmp/longlive-snapshot-probe.OBGzgz/checkout`,
  source3d00d40cb42b9a35fb7823805d91802b5904ddea. Launch wrapper
  `../scripts/run_sprint_streaming957_lane.sh <gpu> <calibration_prompt>`.
- Background publicLongLive2 download exec59353 continues, log
  `results/metrics/sprint24h_20260907/longlive2_assets_fetch_v1.log`.
  Private root `.runtime/models/longlive2-native-bf16`. VAE and reusedT5 are
  checksum-verified;10GB generator was~4.7GB at06:41. Do not restart download.
- The H200 episode-snapshot batch completed6/6, recovered and reviewed; no
  pending work from that job. Check freshInferHub state before new submission.
  The Hpool was busy with other users, so prep briefly returned to queue; this
  was normal. Actual final hardwareH200, notH800.

## Newly closed KV-version experiment — negative promotion

- `episodic_snapshot_probe.py`: bounded6completed rawKV frames before OR after
  native away recache; one later equal-size local-slot replacement at return.
  RawKV gets normal local-relativeRoPE; old coordinates are temporally repositioned.
  First next chunk retains6oldframes, then3, then0 via native eviction.
- CPU guards: immutable copy/no alias, budget before writes, no sink/recent2
  replacement, endpoint counters unchanged, future/duplicate restore rejected.
- GPU39gate3/3 exact pre-return prefixes; the baseline's fullnoise/latent/RGB
  matches previous native official39. Root `episode_snapshot39_gate_v1`.
- Two-seed477batch job
  `zhouhe08__longlive_episode_snapshot_versions_Iter0__3cd428991bea` completed6/6.
  Remote `/kaimm-distill/zhouhe08/longlive-system/outputs/sprint24h_episode_snapshot_3cd4289`.
  Local `results/videos/sprint24h_20260907/episode_snapshot477_h200_v1`.
  All39copied files SHA-match; rsync was absent, so used`cp -a` plus full hash audit.
- Both seeds: full pre78latents identical, sourceframes42–47 and1,725,235,200bytes
  identical across pre/post, fixed local Attention size. All6technicalpass.
- Visual result: pre/post change broad red-object/scene cues but neither restores
  original face/badge/body details. Seed09pre is red/white round-helmet, post is
  red-head/green-round-body; seed10both become wrong red cartoon toys. No stable
  version winner. STOP this local-slot recipe at diagnostic; it is not automatic
  retrieval, not a new quality/Pareto method.
- Do not claim speed from baseline79s versus snapshots62–64s: fixed first-arm
  order/warmup; hashes/copies included. Do not call the version effect pure text:
  construction context and temporal placement are confounds.
- Review `episode_snapshot477_review_v1/INTERPRETATION.md`; source/away/first-return/
  late-return boards were inspected for both seeds. This remains an imperfect
  absence/reappearance stress (unwantedblue toy already appears during away).

## New957system extension (running)

- ExistingFinal route, same common system path, originalRoPE; batch/current-stream
  versus async-priority/current-stream VAE+incremental encode. No history transfer
  overlap enabled. Two development categories, same knownseed20260904,3blocked
  repetitions/arm=12new957executions. Not a repeat of oldformal957.
- First paired results were exact: motion358.50/318.85s, state347.41/300.73s.
  Second pair also exact: motion359.48/317.42, state339.78/299.56. Wait for all
  repetitions; these are same-process blocked repeats, not independent machines.
- Audit once done:
  `python scripts/audit_streaming_length_extension.py --root ../results/videos/sprint24h_20260907/streaming957_local_v1 --output ../results/metrics/sprint24h_20260907/streaming957_local_audit_v1.json`
- Review complete own-video quarters as needed; equivalent arms do not establish
  better absolute quality. Report CPU archive growth; it is still unbounded.

## LongLive2 ready code, assets incomplete, GPU NOT run

- Official pinned submodule `third_party/LongLive2` at
  6b36d20ec6f7958d29d11a704dfa64611a9f2572. Native import and offline tokenizer pass.
- `scripts/fetch_longlive2_reference_assets.py` locks official fullBF16 revision
  8521079b863720a57c1a8d9b19c8d9e6ccb04c0f and Wan assets revision
  921dbaf3f1674a56f47e83fb80a34bac8a8f203e. Download completes only when
  `assets_manifest.json`exists/statuspass; LFS SHA checks are mandatory.
- `scripts/run_longlive2_native_reference.py`: official architecture from_config,
  strict full released generator load (skip fully overwritten20GBbase), noLoRA;
  nativeBF16FA2 only. Native model/scheduler/inference source untouched.
- Encode unique prompts using nativeT5, cache only exact embeddings, offloadT5;
  native`return_latents=True`; freeKV/offloadgenerator before nativeVAE decode.
  Placement/loading are declared, so no native cold-start or cross-model speedup.
- Local gate planned24latents/93pixels, local16/sink8, original704×1280 output.
  Full reference128latents/**509pixels**, local32/sink8,8latentblocks. Schedule
  resampled0/24/48/80 (old1.3B0/21/48/78/120latents). No fair absoluteFPS comparison.
- Native source allocates positive AND negative KV even guidance1; with5B and
  local32 this is~20GBKV, hence local24GB gate explicitlylocal16. No optimization
  of that unused-negative-cache allocation has been implemented or tested.
- Afterassets done and one localGPU free, freeze current runner into a newtemp
  worktree; use`run_on_free_gpu.py`with escalation. Example:
  `/usr/bin/python3 scripts/run_longlive2_native_reference.py --gate --source /home/zhouhe08/MyProjects/0904-longlive-system/publish_repo/third_party/LongLive2 --assets /home/zhouhe08/MyProjects/0904-longlive-system/.runtime/models/longlive2-native-bf16 --output <new gate root> --seed 20260904`
  Supply the existing privatePython overlay inPYTHONPATH, offlineHF flags,
  OMP/MKL2, PYTHONDONTWRITEBYTECODE1. Sourceasserts exactupstreamSHA andFA2.
  If startup fails, summary is now written even for preflight/import failures.
- Only after realgate success, tests/dry-run/push, stage verified assets in new
  privateInferHub inputroot and submit ONE frozen reference batch. Full4controls
  (original2seeds, duck09, absence10) are conditional, not launched. Native
  5B reference is capability evidence; do not spend remaininghours chasing a
  perfect new model reproduction. No public environment changes.

## Tether existing review closure (no new GPU generation)

- Original source/CF+AE automatic477protocol: bothteapot seeds have
  reference/Tether/neutral; bothcyclist automatic masks fail. Thus8videos,4
  downstream outputs prevented by2mask failures, NOT12successfulvideos.
- Separatemanualoracle cyclist0 recovery hasTether+neutral and reusesreference;
  cyclist1still tracking19.29%, held80.71%, stopsbeforeTether.
- Added missing teapot1 andmanualcyclist0 descriptive boards using existing
  latents/videos. Tea0Tether adds hand/lightarcs; tea1pot leavesview despite
  centeredsubject instruction. Manualcyclist0hasno clear overallwinner.
- `docs/TETHER_LONG_VIDEO_FINDINGS_20260908.md` and
  `results/metrics/tether_fixed477_review_v1/teapot_s1`,
  `results/metrics/tether_manual_subject477_review_v1`.
- Neutral zero-bias splitSDPA also causes long trajectory/retrieval divergence;
  do not assign all differences tosemanticbias. Mask on-policy misalignment is
  a hypothesis, not isolated causation; officialtimeaddressing is another confound.
- No newfairAdaCluster/SVOO/SCOPE ranking and noTether expansion requested.

## Figures, tests, inventory

-3new source-hashed figures at `system_insight_figures_v1/index.html`:
  completeH200factorial/firstpacket, boundedpageproducer/dispatch, KVversions.
  Script`build_sprint_system_figures.py`; firstPNGvisually checked.
- ExistingH20048factorial remains unchanged: combinedmedians1.16–1.20×, negative
  repeats preserved; directRoPE underasync negative in3/4groups. Firstpacket is
  muxedpacket, notclientdisplay. Previousplots under`insight_figures_v1`stay.
- Latest saved fullCPU regression472passed/1skipped:
  `cpu_regression_native_reference_pre_gate_v1.log`. Two newinventorytests passed
  afterward. The initialinventory test found `.st_size()` typo; fixed before any
  inventory execution; no generation affected.
- `audit_sprint_video_inventory.py` retrospective named-cohort inventory includes
  gates and repeated trajectories, NOT232independent scientificexperiments.
  `video_inventory_0635_in_progress.json`: expected232, terminal226, remaining6
  at its snapshot; no unregisteredcohorts/errors. Full payloadhashes not run.
  After957done rerun to a NEWoutput with`--hash-payloads`, noactive flag. If adding
  native5Bcohorts, register them explicitly rather than silently omitting them.
- Still needed: nativegate/conditionalreference, complete957audit/review,
  finalwhole-sprint terminal ledger includingfailed/negative/infrastructure
  outcomes, latestfull-flow/mentor3requirements synthesis, finaldiscussion pack
  before14:21. Do not turn absence of an algorithmwinner into a fake completed
  faster-and-better claim.

Memory registry was used for IO/causal evidence conventions (`MEMORY.md36–39`);
final answer requires one memorycitation block. Memory files were not modified.
InferHub SKILL/README/watchdog template were read fully in this turn; public
environments and otherusers' jobs remain untouched.
