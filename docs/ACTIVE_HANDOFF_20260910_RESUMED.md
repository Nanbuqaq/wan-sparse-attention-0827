# Active continuation after the reporting pause

## Latest override: real output/decode overlap proven; three paired repeats preparing

Native output-thread Nsight session13346 is complete. Actual full output/slots
and official Perfetto parser pass. Window84.865s; CPU output14.474s, of which
13.338s overlaps actual GPU1 kernels (~92.15%). GPU0/GPU1 kernel service remains
74.818/77.242s; cross-device kernel overlap69.038s. Pixel D2H bytes unchanged.
Read NATIVE_PIXEL_COMPLETION_TRACE_RESULTS_20260910.md. This is one mechanism
diagnostic, not repeated speed evidence or one-GPU resource superiority.

Three paired full509 timing repeats (6runs) are being frozen, not launched yet:
inline/thread, thread/inline, inline/thread. Same source/seed/weights/hardware /
strict-init loading/pin cap, no new quality sample. Driver:
workspace-root`scripts/run_native_pixel_completion_repeats_20260910.sh`.
It checks actual-output, CUPTI and parser qualification before starting; failures
cannot be dropped to create a positive summary. Read the repeats registration.
All prior GPU tasks are closed; do not relaunch completed gates or trace.

### Completed gate details

Both pixel gates (sessions29116/5111) have finished. Every253/509 actual latent,
decoded RGB, copy byte and pixel-buffer ownership audit passes. Full matched
pair: inline98.804888s, thread84.594745s (~14.4% lower delivery time); first mux
4.733633/4.842218s, so no first-packet improvement. Producer backpressure
7.488363→0.000215s. Extra pinned output43.254MB is recorded; GPU1 peak unchanged.
See NATIVE_PIXEL_COMPLETION_GATE_RESULTS_20260910.md. Single pairs only, not
repeated production statistics or one-GPU resource superiority.

One equivalent thread-mode Nsight capture has now completed, as above.
Driver:workspace-root`scripts/run_native_pixel_completion_trace_20260910.sh`.
It verifies the gated runtime files are unchanged, audits actual files/slots,
exports real CUPTI and measures CPU encode/GPU1 kernel intersection, then runs
the official Perfetto parser. New analyzer CPU test/full suite719pass/1skip.
The measured overlap is recorded above. Default mode stays inline pending repeat assessment.

### Completed numerical-gate launch provenance

Native pixel completion runtime8f49008c9d6d915cfe0ee32ff1d9d3953e319483 at
`/tmp/longlive-pixel-completion.J7hnUG/checkout`. Both short/full stages have
frozen dry-run plans. Short inline→thread is now active under BOTH physical
GPU locks, tool session29116, now finished. Full thread→inline (session5111)
also finished after the short audit. Do not relaunch either gate.
Driver: workspace-root`scripts/run_native_pixel_completion_gate_20260910.sh`.
Outputs: `results/videos/memory_activation_20260910/native_pixel_completion_gate_v1/`;
audits under matching metrics directory; infrastructure logs under local same ID.

Read-only old Nsight analysis found128encode ranges totaling14.810s, overlapping
GPU0 kernels11.427s but GPU1 kernels0s. This supports the hypothesis, not a new
overlap or speedup result. New gate observations are recorded above.

### Completed hybrid and implementation context

Hybrid sessions30132/57133 are finished,2/2 same-condition/actual-prefix audits
pass. Both selected source40–47, with unchanged2.595GBhistoryH2D. Compared with
text-only, source wood/hardware/shiny cloth look closer on both seeds. Seed25
still closes late; seed26 holds the broad open state in sampled late frames.
The both-seed long-term state gate fails: mixed interaction signal, not a robust
autonomous/SOTA method. Review is closed in
`chest_condition_history509_v1/semantic_review.json`; read the factorial results doc.

New system probe uses a bounded CPU output thread behind native VAE, with two
owned pixel buffers and explicit failure/backpressure handling. The old inline
worker blocked on128output groups totaling15.612s CPU-ready→sink-finished spans.
This targets CPU conversion/hash/encode/mux versus GPU decode overlap, NOT a
PCIe-byte reduction or established GPU overlap. Pool budget:5.407MBinput+
86.508MBoutput<128MiB. Default inline remains; no VAE numerical/layout change.
CPU suite718pass/1skip. Short/full real output-and-buffer-ownership gates are
at the live stages recorded above. Driver:
workspace-root`scripts/run_native_pixel_completion_gate_20260910.sh`.
Both arms use the qualified strict-init loading option; cold-time differences
are not method speedups. Require both physical GPU locks and short pass before full.

### Earlier completed gates

Both strict-init GPU gates are complete (sessions22008/7203 closed): actual
complete latent/decoded RGB exactly match the saved253/509 references. Observed
load spans51.829/53.180s versus old147.576/159.974s are single gate observations,
not repeated/steady-state/Attention speedups. See STRICT_CHECKPOINT_INIT_RESULTS.
Default loading stays reference; existing frozen object protocols remain reference.

The two completed hybrid videos finished the condition x history
2x2 on seeds25/26 using six existing cells. Exact same past-text current condition
as the text-only controls, plus the unchanged causal-recent bank; no retuning,
no forced source indices and no startup-mode change. New explicit
`--chest-hybrid-study` registration avoids relaxing old mixed-mode guards.
CPU suite712pass/1skip, post-metadata targeted19pass. Driver:
workspace-root`scripts/run_chest_condition_history_20260910.sh`.
First full runtime/prefix audit on seed25 preceded seed26. Both are now finished;
quality interpretation is recorded above and in their semantic review.

### Completed initialization work

Startup session37328 is complete:145.258s total,97.318s CPU initializer exclusive
host time,40.089s torch.load,2.340s Module.cuda host span. Read
NATIVE_STARTUP_PROFILE_RESULTS_20260910.md and its result directory. No GPU
forward, load optimization or speedup was performed in that diagnostic.

New disabled-by-default constructor mode `strict_checkpoint_no_parameter_init`
skips only direct nn.Parameter initializer functions before COMPLETE strict
loading, preserves Tensor/view/derived-buffer initialization, then uses the
existing independent inference seed. Frozen object protocols still require
reference constructor mode. CPU suite709pass/1skip; real short/full equality
gates have passed as described above. Driver:
workspace-root`scripts/run_strict_checkpoint_init_gate_20260910.sh`.
Short uses the already saved causal-original64 output; full uses the old
settled-bead native509 Dense output. No new independent quality samples.

### Completed previous stages

Both past-text controls are complete (sessions86854/40513 closed),2/2 actual
pre96/pre381 audits pass. Both render broad open/blue-content state, but source
details change; seed25 develops a late hand/lid-closing action, seed26 is partly
open with altered cloth/framing. Descriptive review is closed in
`chest_past_text_control509_v1/semantic_review.json`; read
CHEST_PAST_TEXT_CONTROL_RESULTS_20260910.md. Not an autonomous memory success,
nor permission to retune phrasing/seeds. All video batches are finished.

Observer-only startup profile finished on GPU1, frozen
`/tmp/longlive-native-startup.Zbaztp/checkout`@93b921178ab2a0308c86c16acf5a1f396466de41.
It changes no loading/init operations and runs no generator forward or video.
Outer driver `scripts/run_native_startup_profile_20260910.sh`; results/metrics
under`memory_activation_20260910/native_startup_profile_v1/`, driver logs under
`results/infrastructure/local/native_startup_profile_v1/`. Do not restart it.
Latest full CPU suite705pass/1skip; later Module.cuda observer change has the
same targeted3-test pass. The later constructor optimization is qualified separately above.

### Completed research details

All chestv2 GPU sessions are finished, not active. The complete reviewer passes
6/6cases (4new+2Dense),2/2 same-decision/byte pairs, missing0. Semantic review is
closed: BOTH original/recent fail open-lid/cloth-inside state on BOTH seeds.
Recent seed26 produces blue exterior fragments, not correct state geometry.
Read CHEST_CAUSAL_MEMORY_RESULTS_20260910.md and semantic_review.json under
`results/metrics/memory_activation_20260910/chest_causal_memory509_review_v2/`.
Do not add seeds or tune the frozen selector/delta to chase a positive.

CPU mass-control session71131 is closed:9original FP32 gates pass, per-query
source mass matches exactly but readout still differs. Denoise residual ratios
0.251–0.659 are NOT fractions of quality explained. This is strictly offline,
fixed-Q teacher analysis. Facts: `position_mass_control_v1/mass_control.json`.

VAE timing session18954 is closed:30pairs, medians9.575923/9.257098s, only3.33%
latency reduction with prior numerical drift. Below registered10%; stop this
branch without video expansion/online adoption. Keep all raw timings and note
that bounded CPU tests overlapped part of the diagnostic.

The completed two-case text diagnostic appended exact past settled-source text
appended only to return prompts, no history KV bank. It tests whether current
explicit state conditioning can realize the geometry, not autonomous memory.
See CHEST_PAST_TEXT_CONTROL_20260910.md. Both actual CLI preflights are in the
CPU suite702pass/1skip. Text-control runtime seed25 finished on GPU1,
tool session86854, frozen `/tmp/longlive-chest-text.LgDtII/checkout`
at67bc9d6e3d92990a56034bb15585eb3a45753c52. Both runtime/second plans passed
actual-CLI dry-run. Both cases and audits are finished; do not restart them.
The frozen driver is workspace-root `scripts/run_chest_past_text_control_20260910.sh`:
first`runtime`seed25 GPU1 plus actual-prefix audit; then`second`seed26 GPU0 only
after that audit passes. Do not label these as a retuned prior memory method.

### Prior launch details (now completed)

V2 runtime822155cdcedf3ce4bfd0e4538ecb7f1841ad6e3a, frozen at
`/tmp/longlive-chest-causal-v2.Nl1Sro/checkout`. First full seed25 original
passed generation and actual Dense pre96/pre381 RGB audit (session26166 closed).
The unchanged selector independently chose frames40–47 at96; archive D2H7.786GB,
history H2D2.595GB. This proves runtime/prefix validity, not quality success.
Remaining lane0 (GPU0,seed25 recent) session76894 and lane1 (GPU1,seed26
recent/original) session28254 have finished. Do not restart. All v2 plans are frozen
under `results/infrastructure/local/chest_causal_memory509_v2/`.

Native Perfetto official parser validation is complete with no ingestion errors,
exact activity counts and SQL-recomputed58.843660559s kernel overlap. See
`NATIVE_PERFETTO_READING_GUIDE_20260910.md` and
`results/metrics/memory_activation_20260910/native_perfetto_parser_validation_v1/`.
No new GPU diagnostic/timing repetition was performed for that validation.

Approximate VAE decoder timing has completed, as above: baseline versus
weights_only, saved16latents,5warmups/30randomized pairs, own-variant RGB checks.
It cannot enter the lossless pipeline. See its timing registration; if <10%
decoder gain, stop the branch for now. Do not claim E2E or quality success from
that component timing. The earlier numerical gate is already complete.

Both v1 full-video lanes (sessions3165/50089) have exited with four retained
`preflight_or_import` failures, before model construction. A second old scenario
whitelist rejected the new chest registration. No method video/latent was
generated, so this is not four quality negatives. Details in
`results/infrastructure/local/chest_causal_memory509_v1/INTERPRETATION.md`.

Correction preserves all previously gated selector/installers byte-for-byte;
the shared causal entry validator now consumes the approved object registration.
An exact-CLI CPU check covers all four seed/position combinations and rejects
envelope; full suite694pass/1skip. New driver
`scripts/run_chest_causal_study_v2_20260910.sh` first runs seed25 original as
`runtime`, audits actual pre96 latent/pre381 RGB, then permits `lane0` (seed25
recent) and `lane1` (seed26 recent/original). V2 progress is recorded above.
Never overwrite/restart v1. Both short gates remain valid and
are reused via frozen adapter hashes, not regenerated.

### Historical v1 launch provenance (completed, not active)

Frozen generation runtime d1e62b987a51137a7564f9d1eef07927e726692b:
`/tmp/longlive-chest-causal.GXRrOY/checkout`. Pre-generation CPU suite685pass/1skip;
after CPU reviewer addition690pass/1skip. GPU1 session82278 finished: BOTH
short positions have actual full latent/decoded-RGB equality pass. Recent also
has identical archive/decision/byte records to the original frozen controller.
Output `causal_position_gate64_v1`, in the20260910 video and metrics directories.
Do not repeat either completed gate.

The four v1 cases were launched on GPU0/1, tool sessions3165 and50089,
then failed preflight as described above. Do not relaunch either lane.
Outer driver `scripts/run_chest_causal_study_20260910.sh` (workspace root):
`lane0`=seed25 original/recent; `lane1`=seed26 recent/original. Both short gates
passed before the two physical-lock launches. Then collect actual prefixes, admission,
bytes and source/return boards using the new `review_chest_causal_memory.py`.
Source40–47/return96 is only an offline expectation, not a selector input or
technical-pass requirement. Same-source/same-byte fairness is checked separately.

VAE v2 runtime fda408c at `/tmp/longlive-vae-layout-v2.FboqLj/checkout` finished
normally (session94430 closed). Baseline exact; all three changed layouts are
nonexact (max0.09375,relative L2~.0020 against BF16 native output). No variant
promoted and no diagnostic-time speedup claimed. See NATIVE_VAE_LAYOUT_RESULTS
and `native_vae_layout_gate16_v2/gate.json`. V1 failed before candidate decode
because of a fingerprint stride issue; preserved, not a layout negative.

New position experiment subclasses the original controller in a separate file.
Keep `native_causal_scene_memory.py` at its prior SHA256
5a1d684f3b862d8b347caae072489d8fd612e0b30f76ce411e42a35eede50f06.
Omitted position CLI keeps the old hash-locked class. Never rewrite old screen
hashes or delete their tests to admit this experiment.

## Completed: both Dense lanes and CPU review

Sessions38639/74053 and review46998 exited normally. All8 technical cases and
all4 actual prefix checks pass. Source/away/return/visible-control boards have
been inspected: chest source is qualified on both seeds, envelope is not.
Chest visible controls also lose state after the later cut. Read
OBJECT_STATE_SCREEN_FINDINGS_20260910.md and the review directory's
semantic_review.json; do not rerun the screen or retune its prompts/seeds.

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
