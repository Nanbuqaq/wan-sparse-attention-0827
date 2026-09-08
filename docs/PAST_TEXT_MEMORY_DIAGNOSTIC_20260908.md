# Cheap past-text memory control — frozen before generation

The privileged oldKV interventions recover some attributes but not detailed
identity. Before engineering a more complex memory method, test a cheaper
information source: attributes already present in a past prompt.

Use native1.3B interactive local12/sink1, zero history archive/onload. Compare
ordinary native recache with the same return prompt plus the original reveal's
appearance clause (red tin robot, square head, yellow triangle, white feet).
Do not repeat reveal actions or inspect Dense/SAM/video features. The clause is
extracted from the earlier given instruction, not inferred from future output.

This is a PRIVILEGED diagnostic: the return boundary and relevant past clause
are selected by the experiment. It is not an autonomous text-memory algorithm,
and the cheap text cue does not encode unprompted/generated identity details or
the actually achieved state of a melting/pouring/painting process.

- Real39gate: both branches, identical noise and all pre-return latents.
- If gate passes, existing two development seeds20260909/20260910,120latent /
  477pixel, two arms each. Run one frozen local batch; no new holdout claim.
- Full native recache remains used in both arms; T5 re-encoding is charged.
- Compare against each trajectory's own reveal, not only intended text. A new
  red square robot is not automatically the same original robot.
- No online/Pareto promotion and no timing claim from an ordered single pair.
  Retain failures. Stronger memory methods should not claim value merely from
  retrieving information that this cheap past-instruction control can supply.

The existing pre/postKV results remain negative; this test does not retune or
replace their recorded outputs.
