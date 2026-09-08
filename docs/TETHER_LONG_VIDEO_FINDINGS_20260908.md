# Tether long-video evidence: effect and reusable mechanisms

This closes the descriptive review of existing source/weight-matched outputs.
No new Tether generation was launched for this review. Exact replication is not
the research objective; the goal is mechanisms for faster, better causal video.

## What actually completed

OfficialCF generator + learnedAE retrieval, noLoRA, pinnedTether source;
explicit loader/codec/continuous-VAE corrections are recorded by the runners.

| Original477 protocol | Result |
|---|---|
| Teapot seed0 | Reference, Tether, neutral split-SDPA all complete477 |
| Teapot seed1 | Reference, Tether, neutral split-SDPA all complete477 |
| Cyclist seed0 | Reference complete; automatic subject/mask gate failed; no automaticTether/neutral |
| Cyclist seed1 | Reference complete; automatic subject/mask gate failed; no automaticTether/neutral |
| Separate manual-oracle cyclist seed0 recovery | Original reference reused; Tether+neutral complete477; tracked100% |
| Separate manual-oracle cyclist seed1 recovery | Original reference reused; tracking19.29%, held80.71%; stopped beforeTether |

Manual boxes are offline diagnostic annotations, never automatic or online
success. Failed masks and unrun downstream arms remain explicit. The12-video
automatic protocol is therefore NOT12successful videos:8were produced and
4downstream videos were prevented by two mask failures.

## Observed effect

The official teapot prompt requires the red ceramic teapot to **remain centered
while the camera moves slowly**.

- Seed0: reference and neutral keep the teapot broadly centered while background
  changes toward plants/windows. Tether keeps a dark background longer, but
  introduces a hand/lid interaction and large bright arcs around the subject.
  This is not a demonstrated overall quality improvement.
- Seed1: Tether moves the pot out of the left edge and ends on plants without the
  teapot; this violates the centered-subject requirement. Reference/neutral also
  vary position/appearance, but the subject remains visible at the sampled tail.
- Manual-oracle cyclist seed0: all three differ substantially in later camera,
  bus and scene evolution; Tether moves toward a tree/plaza view with the subject
  much less visible. The prompt only says a cyclist passes a red bus, so leaving
  view is not by itself an instruction failure. No clear identity-quality winner
  can be established from this one manually initialized trajectory.
- Neutral split-SDPA also diverges strongly over time despite zero added bias.
  For teapot0/1, full latent relativeL2 vs native reference is0.8405/0.6844 for
  neutral and1.1002/0.9127 forTether. These measure trajectory difference, NOT
  absolute quality. Do not attribute all reference/Tether differences to semantic
  bias without the neutral control; retrieval feeds altered latents back online.

Review is assistant descriptive inspection, not blind human preference or a
benchmark score. Whole-video16-frame boards were inspected for all9successful
three-arm trajectories, with per-quarter boards retained for deeper audit.

## Research implications, not post-hoc explanations proven causal

1. Grouping can control different information, but preserving one region does
   not guarantee good scene evolution or subject compliance.
2. The reference-derived mask may become misaligned when the generated path
   changes. This is a hypothesis to test with actual on-policy alignment; source
   time-addressing conventions are an additional confound, not silently fixed.
3. Neutral numerical/backend differences can alter later retrieval and amplify
   over long horizons. Short output fidelity and final semantic quality need
   separate measurements.
4. Use these failures to motivate causal group/update/lifetime policies, not to
   spend the sprint perfecting a two-pass implementation or claim sparse speed
   from `target_avg=.25` (which is a bias target, not25%physicalKV).

No same-condition new AdaCluster/SVOO/SCOPE comparison was generated here;
cross-backbone or historical numbers do not support a Tether ranking.

## Evidence

- `../../results/videos/tether_runtime_fixed477_v2/lane0/terminal.json`
- `../../results/videos/tether_runtime_fixed477_v2/lane1/terminal.json`
- `../../results/videos/tether_manual_subject477_v1/seed0/terminal.json`
- `../../results/videos/tether_manual_subject477_v1/seed1/terminal.json`
- `../../results/metrics/tether_fixed477_review_v1/teapot_s0/summary.json`
- `../../results/metrics/tether_fixed477_review_v1/teapot_s1/summary.json`
- `../../results/metrics/tether_manual_subject477_review_v1/summary.json`
