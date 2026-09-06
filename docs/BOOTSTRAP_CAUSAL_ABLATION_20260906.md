# Preregistered single-candidate bootstrap ablation

Matched trajectory capture at `be00491` reproduced all six `cf5c25f` videos:
initial noise, latent bytes, ordered routes and all240 captures agree. Fixed
cross-route replay at `f7d81b4` then compared both admissions on Dense, Final
and aligned actor trajectories, four layers, startup/late and all five calls.

Observed state L9/latent18 (one candidate frame): aligned/raw output-error ratio
averaged1.0242 on Dense,1.0095 on Final,1.0096 on aligned trajectories. All five
calls regress in each trajectory. Motion L9/startup instead improves. This is
evidence of a specific local weakness, NOT proof it caused the video regression.

Two causal interventions are frozen before running their videos:

- `bootstrap_layer=9`: only L9 uses raw Q/K prototypes when exactly one frame
  is eligible in the coarse candidate set. Everywhere else remains aligned.
- `bootstrap_layer=-1`: all layers use raw prototypes at that same single-frame
  condition, then return to aligned. This tests a general startup explanation.

The method ID is `rope_bootstrap_ablation_history`, not a promoted algorithm.
All70/15/15 tiers, V weight1,25% exact union, cache, physical layouts and seeds
stay fixed. Raw backup prototypes are prepared at archive insertion ONLY for
the eligible layers; the selector never reconstructs them from full candidate
KV. Extra metadata bytes and index preparation are charged. Original KV remains
raw and immutable. Coordinate-space checks fail closed.

Four new39-latent cases: motion/state × two interventions, on the same original
physical GPU per prompt. Raw/aligned controls remain the audited `cf5c25f`
artifacts. Capture L0/L9 at latent18, first call only, to audit the intervention.
Before causal attribution, verify all-layer startup's first sparse chunk matches
the Raw control, and L9-only's first changed Attention inputs match the aligned
control. Any failed equality check is a diagnostic failure, not a method result.
Real GPU gates cover single/multiple candidate branches and eligible/ineligible
layers before the video batch. No formal holdout is used or promoted.

In parallel, an independent9-case H-pool seed replication runs source `f7d81b4`:
three same-GPU Dense/Final/aligned triplets, motion seed20260911, state seeds
20260911/20260912. This deliberately asymmetric exploration does not permit a
pooled cross-category mean or discard the original failing state seed.
