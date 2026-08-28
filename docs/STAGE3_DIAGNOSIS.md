# Stage-3 preliminary clustering and V-aware diagnosis

This document is generated from frozen Stage-2 evidence and Stage-3 sampled captured Q/K/V analysis. GPU validation of the new online routes is still a separate gate.

## Why the basic clustering routes collapsed

The seven K-clustering routes share one materialization pattern: K/V tokens are sorted by K-space labels, then a new fixed-block graph is selected without preserving Original-Block or local/time edges. Different cluster families therefore erase the same stability coverage. Their failures are strongly prompt-dependent, with the gymnast case collapsing across every family, while the conductor case is much less sensitive. This common pattern is stronger evidence than a K=128 versus K=256 explanation.

Across the existing captured screen, attention-mass recall and output relative-L2 have correlation -0.076; recall by itself is not a sufficient output-quality target.

## Does V matter?

At the same 25% block budget, sampled-query mean output relative-L2 is 0.1522 for QK block scoring, 0.1279 for the V-prototype score, and 0.1225 for the offline output-residual oracle. The residual objective improves over QK by 19.5%. V norm alone is much worse (0.5771), so the useful signal is query-conditioned V contribution, not globally large V tokens.

Layer 0 is the dominant worst region at early, middle, and late denoise calls in the sampled capture. The online Stage-3 route therefore protects Block/local coverage everywhere and permits V-aware scoring only in the remote remainder.

## Evidence boundary

The V residual result is an offline upper-bound diagnostic, not a completed online/video result. The deployable online approximation and all 50-step conclusions remain gated on GPU correctness and isolated calibration.
