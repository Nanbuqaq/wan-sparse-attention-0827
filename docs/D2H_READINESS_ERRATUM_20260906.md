# D2H readiness: a future-history-state correctness bug

This finding **supersedes final method-selection conclusions** from the earlier
V-aware calibration. Artifacts and measured system equivalences are preserved;
we do not assert that every old video was affected.

## Reproduced evidence

The forced-eviction gate at `eaed968` compared original GPU K/V, stored CPU K/V,
block prototypes, route SHA, five complete forwards, and real upstream cache
updates. The large Final test found:

| Check | Unsafe legacy | Pooled, ready-waited archive |
|---|---:|---:|
| Stored K/V versus GPU source | exact | exact |
| Current five outputs/routes | identical across modes | identical across modes |
| Frame 7 V prototype versus mean of committed V | max-abs **0.633729** | **0** |

Only the V prototype differed. The current calls consumed frames 1–6, so an
output-only gate missed corrupted **future** routing state in frame 7.
Logs: `results/metrics/pooled_archive_gate/large_diagnostic.log` and
`readiness_failure_audit.json`.

## Cause and correction

The legacy path called `.to('cpu', non_blocking=True)` and immediately computed
CPU V means. These destinations are pinned and asynchronous. Cold
`cudaHostAlloc` overhead can hide the race; warm buffer reuse removes that
accidental delay. GPU-computed K means were ordered on the stream, unlike CPU V
means. This is not a floating-point-tolerance discrepancy.

The required dependency is:

`GPU KV → D2H → ready fence → CPU V prototype → later route decisions`

Commit `995b57a` adds the missing legacy copy-stream fence. The pooled archive
already waits for its D2H-ready event before CPU commit. The repaired large
gate passes for Dense and Final: raw data, prototypes, all outputs and ordered
routes match exactly; canonical V-prototype error is zero. The pooled archive
retains zero pinned KV payload and shares a bounded staging pool.

## Evidence and next-run policy

- Old 44/102-case artifacts and the new pre-fix runs remain read-only evidence.
  Dense prompt screening does not use this CPU V-prototype reduction.
- Prior Dense/Final system pairs still demonstrate their recorded exact-output
  equivalence and observed times, but are not a claim of universal legacy
  correctness or complete host-pinned budget compliance.
- V-aware method ranking is quarantined until corrected calibration. Old state
  capture route reconstruction mismatches are consistent with the risk, but
  missing actual prototypes prevent attributing every such mismatch to it.
- Schema-3 complete captures now save the actual online summaries/prototypes.
  Replay checks saved V prototypes against committed V, and initial noise SHA is
  retained to audit same-seed pairings across workers.
- One corrected eight-case development batch compares Dense, Final, peak and
  count on both prompts under a common bounded system. It is a correctness
  repair/control experiment, not an unreported retry of the old matrix.
- No formal admission or formal held-out video batch is frozen yet. The
  previous cost model remains disabled independently of this bug.
