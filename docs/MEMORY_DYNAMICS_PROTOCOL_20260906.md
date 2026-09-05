# Development memory dynamics and causal pulse protocol

The repaired runtime is the platform for exploration, not a new admission claim.
Formal prompt/configuration selection stays sealed. Old results are not replaced.

## Questions frozen before execution

1. The current RAG baseline chooses at most six historical frames using the
   previous completed latent's average-pooled descriptor. Fine Final admission
   operates only inside that set. Measure coarse revisit/eligibility separately
   from per-head Block64 selection; a fine selector cannot recover an omitted
   frame. Retrieval frequency is not causal importance.
2. Existing Final is `refresh_policy=per_chunk`. Equal executed routes and an
   80% union-cache hit rate cannot establish natural denoising stability. On
   selected chunks, recompute shadow routes with each call's current Q summary
   AFTER the actual Attention output. Never execute a shadow route in this run.
3. A same-checkpoint one-chunk pulse changes coarse retrieval to the oldest,
   newest, or reproducibly random eligible frames. Keep coarse count AND actual
   fine selected-token counts equal. Resume the ordinary policy for two further
   chunks. This is a temporal retrieval intervention, NOT identity/state labels
   or proof that semantic memory roles are necessary.

## Execution

Frozen machine-readable policy: `configs/system/memory_dynamics_probe.json`.
Use only calibration_motion/calibration_state, seed20260904, 120 latent frames.
Before the two-lane development batch, a real 39-latent motion runtime gate
executes all new branches. Each physical GPU is locked. No InferHub CPU-only
allocation. The original complete formal matrix remains unstarted.

- Observe executed coordinates at layers0/1/2/3/19/29 throughout the trajectory.
- Fresh shadow routes at latent30/60/114, all four denoising calls plus the clean
  context commit. Report these five phases separately.
- Complete post-RoPE Q/K/V+exact capture: layers0/19, latent30/114, all five calls.
  Recomputed cached-call Q summaries are explicitly marked diagnostic, not
  misrepresented as the input that selected the executed per-chunk route.
- Pulse checkpoint: before latent30; horizon: three chunks (nine latent frames).
  The reference must exactly reproduce uninterrupted suffix latents, the ordered
  route sequence, AND future committed K/V prototype hashes. Fail closed before
  interpreting any pulse if this gate fails.
- Snapshotted GPU caches are cloned; already committed CPU history is immutable
  shared storage with independent list/dict containers. All branches start with
  identical CPU/CUDA RNG state. Local random pulse selection uses its own RNG.

## Interpretation limits

All measured times include diagnostic work and are NOT speed results. There is
no VAE/video review in this first probe: latent divergence measures sensitivity,
not whether a pulse improves absolute video quality. A single trajectory per
category cannot establish a universally good retrieval/admission policy.

Archive growth is measured from actual stored bytes at committed chunks. A
finite-window non-revisited frame is right-censored, not proven safely evictable.
Layer coordinate similarity never authorizes sharing KV tensors across layers.
Complete teachers stay offline; shadow selections use only causal summaries and
committed prototypes. Negative/failed gates retain their outputs and source SHA.
