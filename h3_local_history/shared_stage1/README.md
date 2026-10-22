# Shared H3 fixed-video-KV rules

These files are unmodified shared training/inference sources owned by the training implementation. SHARED_STAGE1_MANIFEST.json pins their hashes. The inference loader verifies this same contract; no separate inference mask is defined.

A uses fixed text; B permits current text to read past committed video K/V; C uses the A inference rule and its separately trained reconstructed-history checkpoint. Every generated block uses 50 native flow evaluations and one video-only clean commit.

The attention head sharding wrapper slices native Q/K/V projection outputs and gathers current outputs before the native output projection. It preserves the shared split visibility and write/read/eviction code. All distributed ranks must execute the same request.

The reference is RAVEN-H3 commit 692ec211d4cd37a99a04773186fc2bf810430f78. This Diffusers/native-velocity/video-only-history adaptation is not a reproduction of the upstream RAVEN engine or results. No checkpoints, source media, dataset annotations, or prompts are included here.
