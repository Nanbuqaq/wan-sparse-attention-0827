# Distance-matched regional sensitivity: qualified, non-uniform signal

The new CPU-only analysis completed all nine FP32 numerical gates, using the
same verified capture and past-source red-mask definition. Three fixed nonred
draws match source frame and squared-distance bin relative to each query site.

Matching is incomplete: center retains613/685 ROI tokens; upper-left and
upper-right685/685; lower-center only257/685 (37.5%). The foreground subset is
fixed across draws, but differs from the older full-ROI experiment. Consequently
the change in ratio cannot be assigned solely to correcting spatial bias.

First-denoise mean sensitivity ratios are roughly1.57–1.93 across the three
sampled layers. Last-denoise L14 is0.887–0.894: matched nonred deletion is slightly
more influential there. Clean-commit L29 spans5.54–9.71 across draws and is not
an extra denoising step or a video-quality result. All raw ratios and coverage
are retained; no favorable draw/layer is selected.

Conclusion: the earlier2.8–8.9x equal-token result does not establish universal
semantic-state importance. Some regional signal remains on the matchable subset,
but it is stage-dependent and lower-center coverage is poor. Do not promote a
three-role router or new online utility from this result. Independent state
workloads and full-video interventions remain necessary.

Facts: `results/metrics/memory_activation_20260909/region_distance_matched_v1/distance_matched_region.json`.
