# Source-block memory: next bounded causal generation slice

The resident-only candidate completed a full toy trajectory but did not retain
its own source identity. This slice instead holds source/away generation fixed
and sparsifies a genuinely retrieved old source at first return. It reuses the
existing coarse current-text/T5 scene choice; that choice is not new research.

Qualified geometry: native5B BF16 CFG1, local32/global8/pin8, chunk8, linear
scale1 absolute RoPE, explicit toy/bead shot changes, and the gated in-place
cache path. At first return the old pinned source slot must be adjacent to the
global sink. Other geometry fails explicitly. No third-party or protected
controller source file is modified.

Archive the original last8 clean frames of each closed scene. For partial modes,
create reusable group K/V means on GPU and copy compact summaries to CPU at
archive time, including their cost/storage in the declared8GiB archive budget.
Groups are either within-frame flat64 or spatial8x8, with real tails. Source
selection never reads raw unselected candidate KV to compute its scores.

Install source per layer only after its first-return Q exists. The fraction1
control copies the complete source into the same native pinned slots and applies
the baseline's recent_virtual temporal rebinding. Own Q projection is independent
of own historical KV before Attention, so this control must exactly match the
existing immediate whole-source causal baseline in complete generated outputs.

Partial policies use mass/value, contrast/value, or independent deterministic
random ranking; uniform32 real Q representatives are fixed. Contrast subtracts
the selected-source summary mixture. K summaries are rebound to current virtual
positions before scoring; rounding means this is still a proxy. Shared and
per-head selection are explicit separate parameters. Each head selects the same
exact token count, quarter initially, trimming the final ranked group if needed.
Canonical original source order is retained per head. Gather ORIGINAL CPU K/V,
copy into the first n pinned slots, rebind temporal K only, and EXCLUDE remaining
stale pinned slots from Attention. No zero fill or prototype execution.

Selection is frozen for four denoising calls plus clean in the first return
chunk. Native next-chunk pinning/rolling then evicts the old source region;
assert that lifetime transition before dropping the mask. This is a registered
policy, not a claim that recomputed routes are identical. Save all source indices,
source version and temporal binding. Gate per-head gather independently.

Budget denominator is the ONE coarse-selected8-frame source, not all archives.
Report full CPU archive/index storage, actual source and metadata transfers,
GPU temporary space and total Attention density. Whole-source control avoids
unneeded fine-score preparation; partial variants pay all added work.

First gate: compare full-source deferred installation against the existing
causal whole-source controller on the64-latent technical protocol. Then validate
one partial route before original-resolution source-preserving comparisons.
The initial offline grouping signal is layer-dependent and not a layer policy:
matched-size spatial groups improve layer14 in one trajectory, other layers mix.
