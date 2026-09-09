# Constructor-only optimization passes short and full output equality

Frozen runtime05b7ad2. Direct Parameter initializers are suppressed only before
complete strict checkpoint loading; ordinary Tensor/view initializers remain,
and inference noise is independently seeded as before. No upstream source or
weight file is changed. Default constructor mode remains reference.

Both gates pass actual complete latent and decoded RGB equality:

| Gate | Pixel frames | Observed load span | Saved reference load span |
| --- | ---: | ---: | ---: |
| Causal-original technical gate | 253 | 51.829s | 147.576s |
| Native settled-bead full gate | 509 | 53.180s | 159.974s |

Both skip1871 observed direct-Parameter initializer calls. These are individual
gate observations across different cases/cache states, not repeated performance
statistics or a measured storage-bandwidth claim. They are consistent with the
prior startup profile's~97s exclusive CPU initialization cost, but no Attention,
steady-state or algorithm-quality gain is inferred.

This is now a qualified opt-in common bootstrap option for the tested native
architecture. Future frozen protocols can offer it equally to Dense and memory
baselines. Existing chest/text/hybrid registrations keep reference construction;
do not quietly change their case identity or compare unmatched cold times.

Facts: `results/metrics/memory_activation_20260910/strict_checkpoint_init_gate_v1/`
short.json and full.json, with actual-output audits and source/reference SHA.
Both tool sessions22008/7203 are closed. No new independent quality samples.
