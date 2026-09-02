# Formal 477-frame basic matrix audit (2026-09-02)

Experiment commit: `694da9e70e8af4202d75734667af847a0ceaf286`

## Terminal status

- expected cases: 44
- terminal cases: 44
- pass: 38
- fail: 6
- negative after the two-prompt storyboard review: 0
- missing: 0
- terminal audit: pass

The six failures are hardware-scoped RTX 4090 24 GB OOM outcomes:

- QLocal-KMeans8 autoregressive: 2
- AdaCluster autoregressive: 2
- SVG2 autoregressive: 2

They are retained as formal failures and are not interpreted as universal
algorithm failures on larger-memory GPUs.

## Artifact and review gates

Every successful case passed:

- 477/477-frame decode;
- latent shape `[1,120,16,60,104]` and finite-value validation;
- video SHA validation;
- verified backend with zero failed/fallback/NaN calls;
- four-quarter storyboard generation;
- automated freeze/cut/flicker diagnostics.

The automated diagnostics did not flag a freeze, cut, or flicker event in the
38 successful cases. Prompt-level method contact sheets and the quarter
storyboards were reviewed for subject identity, background consistency,
irreversible-state reset, and late-quarter degradation. No additional formal
negative was assigned on these two basic prompts. This does not replace the
four-prompt/two-seed Pareto expansion or the 957-frame long-video review.

## Mixed-hardware boundary

- 21 cases were produced on NVIDIA H200.
- 23 cases were attempted on NVIDIA GeForce RTX 4090: 17 pass and 6 fail.

The mixed matrix is valid for terminal coverage, artifact correctness and
coarse visual screening. Absolute runtimes are not ranked across GPU classes.
Final Pareto candidates require same-GPU Dense/Sparse references or a separate
cross-GPU Dense drift gate.

## Audited artifact hashes

- terminal state audit: `228366eadb4d2d352caec720c09dd5de0e07b3c26d13c431cd82c771d94bf133`
- scored manual review: `e7eb912cd5c94284271f55005ca2a62e73433f674d314331ed055d0edf477bc6`
- hardware provenance: `4bcb904b0d923de78cd59407446835eb5ad70fe4596c1cebb828778db3692903`
- mixed artifact recovery: `0bf63abcdd2aeeb7301f3726bc36616c10a0604fb56f5aef47dc295f1336141d`
- manual-reviewed case states: `7608067f5a718f4f038ca89b33d0359953ee5303fdecb775d0580f2415186303`

The outer `results/` tree remains the experiment fact source. Public code and
this small audit document intentionally exclude model weights and generated
media.
