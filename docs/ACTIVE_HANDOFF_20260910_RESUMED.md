# Active continuation after the reporting pause

## Latest override: both Dense lanes and CPU review have finished

Sessions38639/74053 and review46998 exited normally. All8 technical cases and
all4 actual prefix checks pass. Source/away/return/visible-control boards have
been inspected: chest source is qualified on both seeds, envelope is not.
Chest visible controls also lose state after the later cut. Read
OBJECT_STATE_SCREEN_FINDINGS_20260910.md and the review directory's
semantic_review.json; do not rerun the screen or retune its prompts/seeds.

Native VAE layout adapter/tests and a16-latent native-resolution numerical gate
have been prepared but not yet executed on GPU at this override. Only after a
frozen commit/dry-run should run_native_vae_layout_gate_20260910.sh run. It changes
runtime memory format only, not source/weight files, and does not establish a
speedup from diagnostic timing. CPU tests passed for the new context.

The following “current batch” section is historical launch provenance.

The user resumed paper research. Keep all previous results and background work;
do not return to editing the meeting report unless requested. The ongoing goal
is faster/better causal long video, with fair baselines and complete accounting.

## Completed frozen GPU batch: do not restart

- Runtime snapshot: `/tmp/longlive-object-state-screen.bSYfzq/checkout`
- Commit: `29b038ac7c9db292e31bc50cb1b0ccf6e3c092e0`
- Full CPU suite before freezing: **669 passed, 1 skipped**.
- Config SHA: `85466c8494ef77d4a8ffeb645c6fedc981d2a79b521fe6adaa51fd21bdce9f27`.
- Host driver: `/home/zhouhe08/MyProjects/0904-longlive-system/scripts/run_object_state_screen_20260910.sh`.
- GPU0 / tool session38639: seed20260925, chest_revisit → chest_visible_control →
  envelope_revisit → envelope_visible_control.
- GPU1 / tool session74053: seed20260926, envelope_visible_control →
  envelope_revisit → chest_visible_control → chest_revisit.
- Both lanes passed dry-run and launched under distinct physical GPU locks.
- Output: `results/videos/memory_activation_20260910/object_state_screen509_v1/seed<seed>/<case>/`.
- Driver records/logs: `results/infrastructure/local/object_state_screen509_v1/`.

These are eight new **Dense-only source-feasibility** videos, native5B BF16/FA2,
704x1280,128latent/509pixels,local32,CFG1 positive-only allocation. No memory
intervention or pipeline mode. Revisit and visible pairs share the first48
latents and preserve native cut indices2/6/12. Source must be valid on BOTH
seeds before any memory method; no post-result prompt/seed search.

When BOTH lanes terminate, run once from the frozen snapshot (CPU-only):

```
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
 /usr/bin/python3 scripts/review_object_state_screen.py \
 --root /home/zhouhe08/MyProjects/0904-longlive-system/results/videos/memory_activation_20260910/object_state_screen509_v1 \
 --output /home/zhouhe08/MyProjects/0904-longlive-system/results/metrics/memory_activation_20260910/object_state_screen509_review_v1
```

It verifies actual prefix tensors/RGB and makes source/away/return/whole boards;
it does NOT decide semantic success. Main agent must inspect boards for source
state completion, residual actions, true absence and visible-control drift.
See `OBJECT_STATE_DENSE_SCREEN_20260910.md`. Never rerun a finished lane.

## Completed since the user resumed (all preserved; no reruns)

1. Closed native full-resolution two-GPU correctness: both serial and overlap
   match existing actual full latents and decoded RGB. Single warm-delivery
   observations172.73/98.87s are NOT repeated speedup evidence; two GPUs cost
   resources. Initialization varies substantially and is separate.
   `native_pipeline_full509_audit_v1.json`; runtime444ebcd; sessions87540 ended.
2. Closed equivalent two-device CUPTI diagnostic: both device tracks present,
   complete latent/pixel transfer coverage,58.8437s simultaneous kernel work in
   98.3586s. A302,742-event dual-device Perfetto JSON has structural validation,
   NOT new official trace_processor validation. Runtime d1fe6e8; session42472 ended.
   `results/metrics/memory_activation_20260909/native_pipeline_profile509_v1/audit/`.
3. Completed frame/distance-matched red-region teacher control on existing
   captures: nine FP32 gates pass, no GPU/new video. Lower-center matched ROI
   coverage only37.5%; other sites89.5/100/100%. Last-denoise L14 ratio~0.89,
   so no universal state-region sensitivity claim; three fixed draws retained.
   No three-role router or online utility promoted. Runtime642f4d9, session36860 ended.
   `results/metrics/memory_activation_20260909/region_distance_matched_v1/`.

Read `NATIVE_PIPELINE_FULL_RESULTS_20260910.md` and
`REGION_MATCHED_CONTROL_RESULTS_20260910.md` for evidence limits.

## Authority and resource boundaries

- Do not push or submit new InferHub jobs yet. The earlier publication block
  remains; a new asynchronous question explicitly asked permission to publish
  code/config/docs to `https://github.com/Nanbuqaq/wan-sparse-attention-0827`.
  No affirmative response has arrived at this checkpoint. Do not use another
  transport to bypass this. No weights/videos/raw captures are proposed for push.
- The goal control still reports paused. User has been told that this turn is
  executing, but cross-turn automatic continuation needs the goal control resumed.
  Do not create a replacement goal or falsely mark the research complete.
- Local GPU calls require the approved host route because sandbox CUDA visibility
  and /proc are unreliable. Do not infer unavailable GPUs from sandbox nvidia-smi.
- Preserve the three pre-existing untracked files: two
  SPRINT_CONTINUATION_CHECKPOINT_20260908_0400/0500.md files and
  scripts/audit_retrieval_divergence.py. No reset/clean. No training. Old44/102/957
  results, upstream sources and weights remain read-only. No subagents requested.
- Read UTC with clock/date -u; server log filenames/display often use CST. New
  workday filenames use20260910, while acquisition can begin09-09UTC.

## Useful next research work while the lanes run

Use the existing native two-GPU SQLite for kernel-family/service breakdown;
do not conflate combined self/cross-attention kernels with self-attention alone.
The older1.3B Final39 host-gap profile is a different implementation/backbone.
Native5B measured generator and VAE kernel services are both~75–77s, so bottleneck
claims must follow the actual execution baseline, not reuse an old percentage.

No new region router is justified by local sensitivity alone. New object source
validity and independent causal-memory verification are the next algorithm gates.
