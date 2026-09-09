# A stronger offline control for the past red-region diagnostic

The previous equal-token nonred control did not match spatial distance to Q.
Its 2.8–8.9x sensitivity contrast therefore does not establish semantic state
importance. This follow-up reuses the exact same source pixels, ROI definition,
and nine verified actual Q/K/V/O captures. No new video is generated.

Before seeing the new result, fix three control draws (20260910/11/12) and
squared-distance bins of width16 token-coordinate units. Within each source
frame and distance bin relative to each geometric query site, compare the same
number of red-region and nonred tokens. If a stratum cannot be fully matched,
restrict both sides and report the retained ROI fraction; do not change bins
until the result becomes positive. The matchable ROI subset is identical across
the three draws. Selection never reads Q/K/V values or attention scores.

For every record, first recompute original full FP32 attention and apply the
existing BF16 numerical gate. Then remove the matched ROI or each control and
recompute normalized attention. Report per-site coverage, both sensitivities,
all three ratios, and every layer/denoising stage. Do not select the best draw.

If the contrast collapses, keep a negative boundary on the older interpretation.
If it remains, it supports investigating organized regional memory but does not
establish semantic importance, static-state separation, video quality, or an
online method. Source masks may include falling beads; geometric distance bins
do not exactly match relative RoPE phases or learned feature norms. A second
state category and full-video intervention remain necessary before promotion.

Execution is CPU-only with two Torch threads and mmap of the existing capture.
