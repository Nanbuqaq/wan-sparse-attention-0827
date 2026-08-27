# Wan short-video sparse attention Stage-2 report

Audit status: **pass**.

## Frozen evaluation design

Formal prompts: gymnast_ribbon, skateboard_alley, koi_reflections, orchestra_conductor.
Negative holdouts: fox_snow, glassblower.
All routing parameters were frozen using isolated 50-step calibration videos before the formal suite.

## 25% multi-prompt method table

| base_method_id | method_group | psnr | ssim | lpips | flow_epe | temporal_flicker | routing_p50_ms | kernel_warm_p50_ms | generation_elapsed_s | end_to_end_speedup_vs_dense | actual_density | scheduled_density |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| svg2 | paper | 13.1510 | 0.4936 | 0.5770 | 2.2747 | 0.0798 | 27.3377 | 13.6154 | 282.8561 | 0.9401 | 0.2500 | 0.2501 |
| scope | paper_derived | 12.7157 | 0.4552 | 0.5036 | 2.2394 | 0.0845 | 12.9945 | 13.4401 | 237.7425 | 1.1186 | 0.2500 | 0.2501 |
| token_oracle | oracle_baseline | 12.4584 | 0.4630 | 0.5092 | 2.4821 | 0.0915 | 18.8735 | 13.7271 | 256.8434 | 1.0353 | 0.2500 | 0.2501 |
| block | baseline | 12.3200 | 0.4506 | 0.5139 | 2.4699 | 0.0884 | 2.6779 | 13.4641 | 198.5116 | 1.3395 | 0.2500 | 0.2501 |
| qsort_local8 | layout_baseline | 12.3200 | 0.4506 | 0.5139 | 2.4699 | 0.0884 | 8.0731 | 13.4530 | 216.4752 | 1.2284 | 0.2500 | 0.2501 |
| random | baseline | 12.2817 | 0.4545 | 0.6308 | 2.4367 | 0.0891 | 2.6936 | 13.6892 | 200.8534 | 1.3239 | 0.2500 | 0.2501 |
| svoo | paper | 11.6196 | 0.4549 | 0.6278 | 2.2594 | 0.0683 | 15.2927 | 13.7588 | 246.3572 | 1.0794 | 0.2500 | 0.2501 |
| local_3d | baseline | 9.6761 | 0.3236 | 0.6646 | 2.9807 | 0.1281 | 2.6906 | 13.3979 | 199.5008 | 1.3330 | 0.2500 | 0.2501 |
| spatiotemporal | self_cluster | 7.9209 | 0.3049 | 0.6638 | 2.4805 | 0.0874 | 4.9930 | 13.7751 | 214.0309 | 1.2424 | 0.2500 | 0.2501 |
| radius_adaptive | self_cluster | 7.7673 | 0.2858 | 0.6802 | 2.4859 | 0.0967 | 3.2479 | 13.7419 | 203.0794 | 1.3094 | 0.2500 | 0.2501 |
| hierarchical | self_cluster | 7.7104 | 0.3299 | 0.6554 | 2.4019 | 0.0832 | 6.9933 | 13.4046 | 254.8705 | 1.0433 | 0.2500 | 0.2501 |
| query_metric | self_cluster | 7.6709 | 0.2809 | 0.6893 | 2.2959 | 0.0778 | 3.6478 | 13.4639 | 206.4104 | 1.2883 | 0.2500 | 0.2501 |
| adacluster | paper | 7.5427 | 0.2725 | 0.6601 | 2.5104 | 0.1524 | 15.0624 | 13.4690 | 239.0834 | 1.1122 | 0.2500 | 0.2501 |
| capacity_balanced | self_cluster | 7.2815 | 0.2446 | 0.7162 | 2.2702 | 0.0815 | 5.3275 | 13.7679 | 215.2113 | 1.2356 | 0.2500 | 0.2501 |
| product_quantized | self_cluster | 7.1517 | 0.2756 | 0.7091 | 2.5117 | 0.1002 | 5.4585 | 13.7191 | 214.0675 | 1.2422 | 0.2500 | 0.2501 |
| fixed_k128 | baseline | 7.1077 | 0.2777 | 0.7108 | 2.3693 | 0.0908 | 5.3321 | 13.7710 | 209.0239 | 1.2722 | 0.2500 | 0.2501 |

Key observations:

- SVG2 has the highest four-prompt PSNR (13.151 dB) but is slower than Dense (0.94x).
- SCOPE reaches 12.716 dB at 1.12x and is the strongest paper-derived quality-speed compromise in the main panel.
- Original Block is the fastest strong baseline at 1.34x with 12.320 dB.
- None of the six required clean-room clustering families beats Block in the four-prompt main panel; all are retained as negative results.
- On seed 65537, SVOO and SVG2 are strongest (19.156/19.002 dB), showing substantial seed sensitivity.

## Variable-length kernel comparison

| method | graph_kind | backend | kernel_p50_ms | kernel_p90_ms | planner_p50_ms | kernel_speedup_vs_native | max_relative_l2 |
|---|---|---|---|---|---|---|---|
| svg2 | fixedgraph | varlen_triton_csr | 6.1407 | 6.1927 | 0.9004 | 2.2760 | 0.0004 |
| svg2 | fixedgraph | varlen_triton_native | 13.9760 | 13.9968 | 0.0000 | 1.0000 | 0.0000 |
| svg2 | varlen | varlen_triton_csr | 10.0026 | 10.0701 | 0.9000 | 1.7606 | 0.0003 |
| svg2 | varlen | varlen_triton_native | 17.6102 | 17.6629 | 0.0000 | 1.0000 | 0.0000 |
| svoo | fixedgraph | varlen_triton_csr | 5.7307 | 5.7434 | 0.8923 | 2.3623 | 0.0004 |
| svoo | fixedgraph | varlen_triton_native | 13.5375 | 13.5606 | 0.0000 | 1.0000 | 0.0000 |
| svoo | varlen | varlen_triton_csr | 8.7498 | 8.7687 | 0.8972 | 2.0411 | 0.0003 |
| svoo | varlen | varlen_triton_native | 17.8595 | 17.8788 | 0.0000 | 1.0000 | 0.0000 |

Independent 50-step backend videos are retained as end-to-end evidence, but their route graphs diverge after layer 0 and they are not used for pure-kernel ranking.
| base_method_id | psnr | ssim | lpips | kernel_warm_p50_ms | generation_elapsed_s | scheduled_density | padding_ratio |
|---|---|---|---|---|---|---|---|
| svoo_fixedgraph_csr | 10.0355 | 0.3556 | 0.6603 | 16.3431 | 252.0564 | 0.2501 | 0.0005 |
| svg2_fixedgraph_csr | 14.9238 | 0.5027 | 0.5732 | 16.6136 | 296.9212 | 0.2501 | 0.0005 |
| svoo_varlen_csr | 8.0587 | 0.2525 | 0.6876 | 27.0968 | 258.3571 | 0.4302 | 0.4189 |
| svg2_varlen_csr | 14.3759 | 0.5776 | 0.4867 | 29.4790 | 299.7506 | 0.4498 | 0.4440 |
| svoo_fixedgraph_native | 9.9586 | 0.3546 | 0.6395 | 38.4426 | 316.8032 | 0.5002 | 0.5002 |
| svg2_fixedgraph_native | 15.5989 | 0.5990 | 0.5124 | 38.7177 | 360.5807 | 0.5002 | 0.5002 |
| svoo_varlen_native | 8.2073 | 0.2515 | 0.6944 | 52.4787 | 327.0597 | 0.6700 | 0.6268 |
| svg2_varlen_native | 14.4188 | 0.5784 | 0.4827 | 53.7850 | 381.9026 | 0.6761 | 0.6302 |

The first SVOO true-varlen CSR attempt failed because a zero-size padding Q cluster had no active edge. The planner was corrected to require edges only for non-empty Q clusters; the failure JSON is archived and the rerun completed successfully.

## Correctness and numerical boundary

Strict numerical status: **fail**; classification: `route_and_kernel_attention_correct_but_multilayer_bf16_accumulation_exceeds_strict_latent_gate`.
Maximum direct Attention relative L2: 0.003139.
Maximum one-step latent relative L2: 0.028859.
The report does not describe a failed strict latent gate as byte-level or numerical equivalence.

## K256 negative recheck

K256 classification: **negative_holdout**; Stage-2 PSNR delta versus K128: -0.0081 dB.
K256 remains an absolute collapse, but K128 collapses similarly under the Stage-2 backend; the K-specific failure is not reproduced.
The original Stage-1 collapse remains preserved independently; K128 stays in the required baseline table as an explicit negative result.

## Case-level statistics

Bootstrap and Holm calculations use complete prompt/seed videos as samples; the 81 frames are not independent observations.
| reference | method | cases | psnr_delta_mean | psnr_ci_low | psnr_ci_high | wins | ties | losses | worst_case_delta | holm_p | interpretation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| block | svg2 | 4.0000 | 0.8311 | -0.2831 | 1.9452 | 2.0000 | 1.0000 | 1.0000 | -0.5324 | 1.0000 | insufficient_evidence_to_distinguish |
| block | scope | 4.0000 | 0.3957 | 0.1239 | 0.8280 | 4.0000 | 0.0000 | 0.0000 | 0.1130 | 1.0000 | insufficient_evidence_to_distinguish |
| block | token_oracle | 4.0000 | 0.1384 | -0.0082 | 0.3664 | 2.0000 | 2.0000 | 0.0000 | -0.0376 | 1.0000 | insufficient_evidence_to_distinguish |
| block | qsort_local8 | 4.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4.0000 | 0.0000 | 0.0000 | 1.0000 | insufficient_evidence_to_distinguish |
| block | random | 4.0000 | -0.0383 | -0.9829 | 1.1820 | 1.0000 | 0.0000 | 3.0000 | -1.2471 | 1.0000 | insufficient_evidence_to_distinguish |
| block | svoo | 4.0000 | -0.7004 | -2.7137 | 0.7009 | 2.0000 | 0.0000 | 2.0000 | -3.7719 | 1.0000 | insufficient_evidence_to_distinguish |
| block | local_3d | 4.0000 | -2.6439 | -4.1440 | -1.1438 | 0.0000 | 0.0000 | 4.0000 | -4.8563 | 1.0000 | insufficient_evidence_to_distinguish |
| block | spatiotemporal | 4.0000 | -4.3991 | -11.2193 | -0.7036 | 0.0000 | 0.0000 | 4.0000 | -14.6737 | 1.0000 | insufficient_evidence_to_distinguish |
| block | radius_adaptive | 4.0000 | -4.5527 | -11.2168 | -0.5781 | 0.0000 | 0.0000 | 4.0000 | -14.6997 | 1.0000 | insufficient_evidence_to_distinguish |
| block | hierarchical | 4.0000 | -4.6096 | -11.3249 | -0.8342 | 0.0000 | 0.0000 | 4.0000 | -14.6851 | 1.0000 | insufficient_evidence_to_distinguish |
| block | query_metric | 4.0000 | -4.6490 | -11.0915 | -0.6060 | 0.0000 | 0.0000 | 4.0000 | -14.5249 | 1.0000 | insufficient_evidence_to_distinguish |
| block | adacluster | 4.0000 | -4.7773 | -11.1645 | -0.4060 | 0.0000 | 0.0000 | 4.0000 | -14.6916 | 1.0000 | insufficient_evidence_to_distinguish |
| block | capacity_balanced | 4.0000 | -5.0385 | -11.1978 | -0.5763 | 0.0000 | 0.0000 | 4.0000 | -14.6799 | 1.0000 | insufficient_evidence_to_distinguish |
| block | product_quantized | 4.0000 | -5.1682 | -11.5231 | -1.2423 | 0.0000 | 0.0000 | 4.0000 | -14.5402 | 1.0000 | insufficient_evidence_to_distinguish |
| block | fixed_k128 | 4.0000 | -5.2122 | -11.2833 | -0.7641 | 0.0000 | 0.0000 | 4.0000 | -14.6797 | 1.0000 | insufficient_evidence_to_distinguish |
| fixed_k128 | svg2 | 4.0000 | 6.0433 | 1.8183 | 11.4987 | 4.0000 | 0.0000 | 0.0000 | 0.4000 | 1.0000 | insufficient_evidence_to_distinguish |
| fixed_k128 | scope | 4.0000 | 5.6079 | 1.3501 | 11.5898 | 4.0000 | 0.0000 | 0.0000 | 1.2073 | 1.0000 | insufficient_evidence_to_distinguish |
| fixed_k128 | token_oracle | 4.0000 | 5.3506 | 0.8210 | 11.6614 | 4.0000 | 0.0000 | 0.0000 | 0.4674 | 1.0000 | insufficient_evidence_to_distinguish |
| fixed_k128 | block | 4.0000 | 5.2122 | 0.7641 | 11.2833 | 4.0000 | 0.0000 | 0.0000 | 0.4338 | 1.0000 | insufficient_evidence_to_distinguish |
| fixed_k128 | qsort_local8 | 4.0000 | 5.2122 | 0.7641 | 11.2833 | 4.0000 | 0.0000 | 0.0000 | 0.4338 | 1.0000 | insufficient_evidence_to_distinguish |
| fixed_k128 | random | 4.0000 | 5.1739 | 1.0311 | 11.3855 | 4.0000 | 0.0000 | 0.0000 | 0.2435 | 1.0000 | insufficient_evidence_to_distinguish |
| fixed_k128 | svoo | 4.0000 | 4.5118 | 1.0188 | 8.6896 | 3.0000 | 1.0000 | 0.0000 | 0.0023 | 1.0000 | insufficient_evidence_to_distinguish |
| fixed_k128 | local_3d | 4.0000 | 2.5683 | -1.4457 | 7.4268 | 3.0000 | 0.0000 | 1.0000 | -2.9978 | 1.0000 | insufficient_evidence_to_distinguish |
| fixed_k128 | spatiotemporal | 4.0000 | 0.8131 | -0.0558 | 2.3457 | 2.0000 | 1.0000 | 1.0000 | -0.1174 | 1.0000 | insufficient_evidence_to_distinguish |
| fixed_k128 | radius_adaptive | 4.0000 | 0.6596 | 0.0129 | 1.7261 | 2.0000 | 2.0000 | 0.0000 | -0.0200 | 1.0000 | insufficient_evidence_to_distinguish |
| fixed_k128 | hierarchical | 4.0000 | 0.6026 | -0.1100 | 1.9158 | 1.0000 | 2.0000 | 1.0000 | -0.1499 | 1.0000 | insufficient_evidence_to_distinguish |
| fixed_k128 | query_metric | 4.0000 | 0.5632 | 0.0838 | 1.3751 | 3.0000 | 1.0000 | 0.0000 | 0.0128 | 1.0000 | insufficient_evidence_to_distinguish |
| fixed_k128 | adacluster | 4.0000 | 0.4349 | -0.0806 | 0.9505 | 2.0000 | 1.0000 | 1.0000 | -0.1493 | 1.0000 | insufficient_evidence_to_distinguish |
| fixed_k128 | capacity_balanced | 4.0000 | 0.1737 | 0.0163 | 0.3311 | 2.0000 | 2.0000 | 0.0000 | -0.0002 | 1.0000 | insufficient_evidence_to_distinguish |
| fixed_k128 | product_quantized | 4.0000 | 0.0440 | -0.9984 | 1.1708 | 2.0000 | 0.0000 | 2.0000 | -1.3777 | 1.0000 | insufficient_evidence_to_distinguish |

## Evidence limits

PSNR/SSIM/LPIPS/Flow/flicker measure fidelity to the matched Dense run, not absolute aesthetic quality. Dense prompts were therefore frozen using a separate normal-step visual review. Non-significant results are reported as insufficient evidence to distinguish methods, not as equivalence.
