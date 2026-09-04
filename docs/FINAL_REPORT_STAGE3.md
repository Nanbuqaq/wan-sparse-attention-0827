# Wan short-video sparse attention Stage-3 final report

Audit status: **pass** (19/19 checks).

> **SVG2 clarification (2026-09-04):** the `svg2` rows in this frozen report
> use an exact-density adaptation without the upstream Dense guard, not the
> Wan-14B/720p Top-p policy. A same-prompt unified-varlen 2x2 later showed that
> exact-25% collapses with or without the guard, while Top-p passes with or
> without it. See `docs/SVG2_DEBUG_2026-09-04.md`. The historical numbers and
> 7/7 visual failures below remain unchanged.

## Outcome

Stage-3 produced a usable basic clustering route, an online V-aware route, and a final Block+local+cluster/V hybrid. All three preserve original token order and execute original Q/K/V. At 25% actual Q-K pair density, each of these three Stage-3 new methods generated normal recognizable videos for all four formal prompts, the second seed, and both negative cases; none showed the subject-disappearance or large-white-region collapse of the cluster-only routes.

## Four-prompt 25% main ranking

| base_method_id   |   psnr_mean |   ssim_mean |   lpips_mean |   flow_epe_mean |   temporal_flicker |   routing_p50_ms |   kernel_warm_p50_ms |   generation_elapsed_s |   end_to_end_speedup_vs_dense |   actual_density |
|:-----------------|------------:|------------:|-------------:|----------------:|-------------------:|-----------------:|---------------------:|-----------------------:|------------------------------:|-----------------:|
| svg2             |     13.1510 |      0.4936 |       0.5770 |          2.2747 |             0.0798 |          27.3377 |              13.6154 |               282.8561 |                        0.9401 |           0.2500 |
| scope            |     12.7157 |      0.4552 |       0.5036 |          2.2394 |             0.0845 |          12.9945 |              13.4401 |               237.7425 |                        1.1186 |           0.2500 |
| stage3_hybrid    |     12.6896 |      0.4550 |       0.5151 |          2.3691 |             0.0893 |           8.4381 |              13.7568 |               220.1394 |                        1.2079 |           0.2500 |
| coverage_cluster |     12.6364 |      0.4659 |       0.5177 |          2.4905 |             0.0922 |           7.3019 |              13.7587 |               219.2591 |                        1.2128 |           0.2500 |
| vaware_cluster   |     12.6031 |      0.4572 |       0.5101 |          2.4480 |             0.0899 |           8.1961 |              13.7585 |               220.3206 |                        1.2070 |           0.2500 |
| block            |     12.3200 |      0.4506 |       0.5139 |          2.4699 |             0.0884 |           2.6779 |              13.4641 |               198.5116 |                        1.3395 |           0.2500 |

The final hybrid reaches 12.690 dB and 1.208x Dense speed. It is +0.370 dB above Block with 4 wins and 0 losses, while retaining a real speedup. It is only -0.026 dB from SCOPE but is faster (1.208x versus 1.119x).

The complete-case bootstrap CI for hybrid minus Block is [+0.144, +0.599] dB. The exact case-level sign-flip test has Holm-adjusted p=0.625; with four formal cases, this is reported as insufficient evidence for a significance claim, not equivalence.

## Why the previous basic clustering collapsed

Fixed-K and the six Stage-2 clustering families all sorted K/V by K-space labels and then selected a fresh fixed-block graph. They did not replace V with a centroid, but they removed the original Block/local/time connections. Seven different clustering families failed on the same prompts, so the shared materialization and missing coverage are more important than K=128 versus K=256 or a particular K-means variant.

On 12 captured Q/K/V points spanning layers 0/9/19/29 and early/middle/late denoise calls, Fixed-K128 output relative-L2 is 0.338; the old clustering routes range from 0.261 to 0.383. Block is 0.161, while stable coverage is 0.159. Layer 0 is the dominant worst region.

The frozen usable basic route reserves 70% of each row budget for Original-Block edges, 15% for local/time coverage, and 15% for remote cluster retrieval. In the four-prompt panel it is +0.316 dB versus Block (3 wins, 1 loss) at 1.213x Dense speed.

## What V contributes

At the same 25% block budget, sampled-query QK block scoring has output relative-L2 0.1522. V-prototype scoring reduces it to 0.1279; the offline output-residual oracle reaches 0.1225. V norm alone is poor (0.5771). The useful signal is query-conditioned V contribution, not globally large V tokens.

The online prototype route is +0.283 dB versus Block (3 wins, 1 loss) at 1.207x. The final hybrid uses the stronger residual approximation only for the remote remainder; it cannot evict Block/local guarantees.

## 100% backend and latent separation

| base_method_id   |   psnr_mean |   ssim_mean |   lpips_mean |   flow_epe_mean |   temporal_flicker |   generation_elapsed_s |   end_to_end_speedup_vs_dense |   kernel_warm_p50_ms |   planner_p50_ms |
|:-----------------|------------:|------------:|-------------:|----------------:|-------------------:|-----------------------:|------------------------------:|---------------------:|-----------------:|
| block_100_csr    |     27.2001 |      0.9572 |       0.0460 |          0.7164 |             0.0112 |               353.3700 |                        0.7598 |              63.8277 |           0.9118 |
| block_100_fixed  |     28.2681 |      0.9599 |       0.0481 |          0.6809 |             0.0101 |               318.4850 |                        0.8430 |              52.2838 |           0.0000 |

Dense repeat latent noise is 0.000000. At 100% density, fixed64 one-step latent relative-L2 is 0.0226 and Stage-3 fixed/CSR are 0.0226/0.0226. All routes execute with exact 100% pairs and no fallback, but strict 1% latent equivalence fails. The 25% video error therefore contains a non-negligible multilayer BF16/backend component and cannot be attributed only to sparse retrieval.

## Kernel decision

On one identical Stage-3 RoutePlan, fixed64 costs 4.724 ms. The best CSR setting costs 5.984 ms including 0.892 ms planning. CSR is retained as a correct negative result but is not the final backend. Low-frequency cluster refresh remains active, and original-order execution eliminates permutation/inverse overhead.

## Robustness and absolute visual quality

Second-seed results:

| base_method_id   |   psnr_mean |   ssim_mean |   lpips_mean |   flow_epe_mean |   temporal_flicker |   end_to_end_speedup_vs_dense |
|:-----------------|------------:|------------:|-------------:|----------------:|-------------------:|------------------------------:|
| svg2             |     19.0024 |      0.5878 |       0.3882 |          1.1998 |             0.0207 |                        0.9372 |
| block            |     15.9164 |      0.6693 |       0.3651 |          1.3567 |             0.0219 |                        1.3322 |
| scope            |     15.8491 |      0.6301 |       0.3580 |          1.2943 |             0.0191 |                        1.1217 |
| vaware_cluster   |     15.7480 |      0.6295 |       0.3796 |          1.4555 |             0.0234 |                        1.2039 |
| stage3_hybrid    |     15.7308 |      0.6288 |       0.3729 |          1.4443 |             0.0240 |                        1.1760 |
| coverage_cluster |     14.6414 |      0.5994 |       0.3955 |          1.4926 |             0.0254 |                        1.2035 |

Negative-case averages:

| base_method_id   |   psnr_mean |   ssim_mean |   lpips_mean |   speedup |
|:-----------------|------------:|------------:|-------------:|----------:|
| svg2             |     13.3924 |      0.5394 |       0.4345 |    0.9366 |
| scope            |     12.4883 |      0.5126 |       0.4795 |    1.1251 |
| coverage_cluster |     12.3560 |      0.5272 |       0.4673 |    1.2217 |
| stage3_hybrid    |     12.3359 |      0.5227 |       0.4688 |    1.2134 |
| vaware_cluster   |     12.3263 |      0.5224 |       0.4713 |    1.2117 |
| block            |     12.0399 |      0.5154 |       0.4710 |    1.3415 |

Formal visual review marks all 21 Stage-3 new-method cases as normal and collapse-free. SVG2 is numerically high in relative PSNR but visually collapsed in 7/7 reviewed cases. Relative-to-Dense metrics are therefore not treated as absolute video-quality scores.

## LongLive migration

The transferable design is: recent/local tokens receive exact guaranteed coverage; historical tokens are grouped only for retrieval; remote ranking uses cluster relevance plus a query-conditioned V/output-residual proxy; selected historical entries still execute their original KV. Refresh cluster metadata at low frequency, keep the total recent+history pair budget exact per call, and benchmark CSR only on the actual variable history graph. For fixed 64-token short-video graphs, CSR was slower; LongLive should not assume the result transfers to imbalanced history lengths.

A practical LongLive starting allocation is 80% recent/content Block coverage, 10% explicit local/time coverage, and 10% history cluster/V-aware retrieval, with a small early-step shift toward the guaranteed recent budget. This is a migration hypothesis, not a completed LongLive experiment.

## Evidence limits

The main statistical unit is one complete prompt/seed video. Four main cases cannot support strong significance claims after multiple-comparison correction. PSNR/SSIM/LPIPS/Flow/flicker are fidelity metrics to matched Dense output; absolute visual review remains separate. The interrupted second-seed attempt is retained as a BrokenPipe failure and its successful rerun is the ranked result.

## Artifacts

- Frozen suite: `configs/stage3_formal_50step.json`
- Final audit: `results/manifests/final_audit_stage3.json`
- Case metrics and statistics: `results/metrics/stage3_formal_50step/`
- Captured diagnostics: `results/metrics/stage3_qkv_diagnostics/`
- Quality-speed figure: `results/figures/stage3_formal_50step/quality_speed_pareto.png`
- Comparison videos: `results/videos/stage3_comparisons/`
