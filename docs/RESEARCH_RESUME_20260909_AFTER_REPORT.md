# Resume after the mentor-report discussion

The user explicitly resumed the original paper-research objective. The meeting
HTML remains a reporting artifact, not the research execution plan. Its removal
of pipeline pages was presentation scope, not cancellation of system research.

## Verified continuation state

- Local HEAD before this work: 717a8f3. Both 4090s were idle on the host check.
- Existing native_pipeline_gate64_v1 serial/overlap both completed. Their summary
  hashes still match native_pipeline_gate_partial_v1.json, whose status is pass.
  Actual full latents and decoded 253-frame RGB match the native control.
- Do not rerun either small gate or the completed native VAE gates.
- Pre-existing pending changes add robust abandoned-iterator cleanup, separate
  final-delivery timing from later artifact hashing, and NVTX/profile capture.
  The CPU suite before the new geometry tests passed 648 tests, one skipped.
- After deriving audit geometry from the paired reference and adding seven
  regression cases, the full CPU suite passed 655 tests with one skip.
- Keep the three earlier user/unrelated untracked files untouched. No push or
  new InferHub submission without the previously missing explicit publication
  authorization. No training or changes to old result directories.

## Next bounded experiment: native full-resolution pipeline correctness

Two conditions, same two physical GPUs: serial and bounded overlap. Use seed
20260919, settled_bead_revisit, CFG1 positive-only KV allocation, local32,
fixed native adaLN16/stage1, native BF16 FA2 and native Wan VAE. No memory
intervention. Reference is the already-generated original Dense trajectory:

results/videos/memory_activation_20260909/settled_state_screen509_v1/lane0/settled_bead_revisit

Each run is 128 latent / 509 pixels at 704x1280. Test actual full latent equality,
raw-RGB and decoded RGB equality, shot/prompt identity, bounded input/output
buffers, frame continuity, measured byte counts and independent device peaks.
Failure is retained and prevents performance promotion; it does not justify
relaxing equivalence or silently changing VAE.

No single paired run establishes a speedup. Model/T5 initialization and VAE
placement are separate recorded costs. Pipeline complete_s is warm delivery
from the generation start, not cold end-to-end including all initialization.
Real CUPTI multi-device timelines follow only after correctness passes. Fair
performance work must also compare resource use, not only two-GPU vs one-GPU
latency. The prior small-case time difference is not a paper result.

## System characterization prompted by the discussion

The Final39 instrumented GPU-idle total has been partitioned by the innermost
CPU NVTX scope. That identifies where gaps occur, not why each CPU instruction
costs time. Roughly 9.40 s remains inside un-subdivided self-attention scope;
do not label all of it Python or CPU-DRAM saturation. The 56.54 s diagnostic
and 46.20 s later control are not a clean profiler-overhead factorial.

Next analysis should separate instrumentation, host dispatch, CPU packing,
metadata and blocking API behavior with controlled sampling. Do not pool
host-inclusive ranges, GPU service sums and critical-path exposure. Existing
sum-to-idle records are reusable; do not regenerate videos just to rebuild them.

Algorithm direction stays open: state-memory activation has a reproducible
conditional position-binding signal, but action contamination/identity failures
remain. Blue-canvas source feasibility failed; no new sparse method may claim
success there. Generic pipelines do not replace a causal-memory mechanism.
