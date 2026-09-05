# Memory exploration: observations, falsifications and one runtime candidate

The full LongLive co-design plan remains unfinished. This report concerns one
development exploration round, not a formal result or a LongLive2 comparison.
Old matrices and formal holdouts are untouched. No training or new external model.

## What was actually run

- `d6b20e4`: one real motion39 gate and motion/state120 diagnostics. Each forks
  reference/oldest/newest/random from latent30 for three chunks. No-op latent,
  ordered route and archived-prototype equality passes. Both120 initial-noise
  SHAs match. 552 route observations and40 complete Attention captures in the
  two120 cases; the39 gate adds66 observations and10 captures.
- `79b83a3`: 40 full-teacher call evaluations; eight actual153-frame pulse videos
  reconstructed from the common30-latent prefix and generated9-latent suffix.
  No baseline future is spliced after a fork. All common raw pixel prefixes match.
- `cc8f2e9`: eight equal-budget representation×refresh factorials and four
  same-candidate-set permutation audits, each with original/sorted/reverse/rotate.
- `9e16f23`: independent causal `rope_aligned_final_history` runtime candidate;
  small/large real forward, forced-eviction, raw-KV/prototype and FP32 gates pass.
- `cf5c25f`: six matched local cases, Dense/Final/aligned × motion/state,39 latent
  /153 pixel frames. Same GPU per prompt, same generic optimized system, identical
  initial noise. All6 pass technically, no fallback, missing0. Relative quality
  was evaluated with locked LPIPS weights/versions; this is not absolute quality.

## Observations that change the research direction

1. **Two retrieval levels.** RAG picks at most six frames before fine Block64
   admission. Fine routing cannot recover an omitted frame. At120,100 frames
   have been eligible;59/42 were visited in motion/state. Unvisited frames and
   last revisit intervals are right-censored, not proved safely evictable.
2. **No free layer reuse.** Median four-layer0–3 adjacent token Jaccard is0.216
   (motion)/0.311(state), well below0.8. Per-layer fine-block revisits can occur
   after19/23 chunks. A tiny permanent hot set or a short recency horizon is not
   established by these data. No cross-layer KV sharing is authorized.
3. **Cache-enforced equality is not natural route stability.** Recomputed raw-Q
   routes differ, especially at clean commit. But fresh routing can worsen the
   actual full-context Attention error: motion L0 latent30 clean,0.0641→0.1065.
   Therefore route change alone did not justify commit-time refreshing.
4. **Representation beats extra refreshing in the sampled clean calls.** The
   equal-token2×2 separates raw/aligned prototypes from first/current Q. Aligned
   first-route reuse improves all8 sampled clean outputs,1.75–16.28%, median3.27%.
   Refresh timing has mixed effects. These are correlated points from two
   trajectories, not eight independent experiments or full-video evidence.
5. **Order is a necessary control.** The state `newest` pulse has the SAME six
   frames as the reference, reordered. It is not an age intervention. In all
   four same-Q captures, all three permutations preserve logical edges. FP32
   differences are about1.2e-7 relative-L2; BF16 FA2 differences are1.35e-4–3.35e-4
   (max-abs0.015625). The state order-only pulse later diverges appreciably, but
   these four captures do not prove every later route stayed identical. Future
   age tests need fixed victim/replacement counts and order-only controls.
6. **Bounded staging is not bounded streaming storage.** Actual CPU KV rises
   by287,539,200 bytes per additional latent frame, reaching31,054,233,600 bytes
   at120. Pinned KV is zero. Diagnostics include capture overhead: no speed
   extrapolation or steady-state latency claim follows from their wall times.

## Runtime candidate: mixed video result, not promoted

Same physical4090 per prompt, same source and initial-noise SHA:

| Prompt | Method | LPIPS vs Dense | Late-quarter LPIPS | Latent relative-L2 | Complete time(s) |
|---|---|---:|---:|---:|---:|
| motion | Final |0.09156|0.25874|0.29827|57.716|
| motion | aligned-first |0.07563|0.21267|0.26704|56.401|
| state | Final |0.05843|0.13084|0.23054|57.546|
| state | aligned-first |0.06246|0.15029|0.24588|57.731|

Dense complete times:63.419/61.743s. These are single, ordered development
measurements, not a repeat-controlled speedup estimate. Sparse history admission
is25% in both methods; with cache, aggregate H2D is5% of repeated candidate bytes
(Dense20%). Do not confuse these denominators.

The candidate improves motion but regresses state, so it fails the preregistered
two-category gate. Capture-level8/8 does not authorize formal promotion. Keep
the candidate, code and mixed evidence for exploration; do not silently replace
legacy Final. Full denoising-step replay is the next bounded diagnostic, before
claiming the cause is solely long-horizon feedback or adding a layered heuristic.

Assistant inspection of all-six16-frame overviews: subject/container persists
without an obvious cut at these sampled frames; the motion alternatives differ
in toy stance and ball timing, and the state alternatives in reflections/inflow
and cup details. The cup fills then approaches a plateau. No absolute state or
identity advantage is established by that review. Quarter storyboards are saved;
overview inspection is not a human full-video quality label.

## Evidence and figures (outer results root)

- `metrics/memory_exploration_report_20260906/`:3 figure families, CSV and35/35
  artifact audit, fail0/missing0. This audit predates the separate6-case video
  batch and does not mean the entire project is complete.
- `metrics/memory_dynamics_d6b20e4/`: original observations, complete QKV and forks.
- `metrics/proxy_order_followup_cc8f2e9/`: factorial and FP32/BF16 order controls.
- `metrics/clean_phase_proxy_79b83a3/`: precursor clean-call phase analysis.
- `metrics/aligned_final_gate_9e16f23/`: real runtime gates.
- `videos/aligned_final_cf5c25f_local/`: all-six code/config/log/video/latent/states
  and terminal audit. Quality:`metrics/aligned_final_quality_cf5c25f/`.
- `metrics/aligned_final_storyboards_cf5c25f/`: six overviews and24 quarter boards.

Remaining original work includes longer/independent development validation,
causal role/lifecycle interventions, verified prefetch and conditional kernels,
fair LongLive/LongLive2 provenance, then frozen formal477/957 evaluation. There is
no new admission, role method, or adaptive dataflow promoted into the holdouts.
