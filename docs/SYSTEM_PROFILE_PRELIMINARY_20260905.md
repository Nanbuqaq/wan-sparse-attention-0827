# LongLive system preliminary profile

## Scope

This report covers read-only capture analysis and isolated replay only. It does
not promote a video method or claim end-to-end speedup. New sparse videos remain
blocked until the independent RAG-Dense state prompt screen is reviewed and
frozen.

## Capture layout structure

Twelve RAG-Dense captures span layers 0/9/19/29 and early/middle/late history
points. Relative to exact compact transfer, mean physical tradeoffs are:

| Layout | Copied bytes | Source-run reduction |
|---|---:|---:|
| Block64 | 1.1003x | 8.18% |
| Page256 | 2.4217x | 43.42% |
| Frame1560 | 4.0000x | 85.05% |

Thus continuity is not free: Page/Frame layouts sharply reduce runs by moving
substantially more K/V. Logical Attention edges remain unchanged in every row.

## H200 transfer replay

An eight-GPU frozen batch at commit `83fb91a1d8eb2e1384ef2a0bf0b74757e8aee080`
completed 8/8 captures, 16 layout/execution cases per capture, 5 warmups and 30
measurements. Every lane self-reported `NVIDIA H200`, compute capability 9.0.

Direct multi-run used a candidate K/V tensor pinned once outside the timed
replay. Under that assumption, Frame1560 direct was fastest on 7/8 captures and
Page256 direct on 1/8. On one early exact case, direct multi-run took about
2.18 ms for 62 K/V copies, whereas packed separate achieved about 20 GB/s H2D
service time (0.12 ms) but paid roughly 2.9 ms CPU gather/pack, making the total
about 3.0 ms. Higher copy bandwidth therefore did not by itself reduce the
whole materialization path.

The one-time full-candidate pin cost had a median of 1.85 seconds across the
eight captures. The real archive is not yet proven to keep the necessary source
pages pinned within the configured host budget. Direct coarse layouts are
therefore not promoted until archive D2H, pinning, and exposed end-to-end wait
are included.

## Cost-model gate

Six captures calibrated the initial non-negative bytes/run/copy model; two
held-out captures audited it. Held-out MAPE was 37.40%, above the frozen 15%
gate. `cost_aware_admission_allowed=false` is therefore final for this model
version. Set-marginal utility candidates are stopped rather than evaluated with
an inaccurate predictor; static candidates may continue.

## Query membership capture

A four-GPU b300 capture batch compared identical 25% physical unions. Mean
history-pair density and worst history-only relative-L2 error were:

| Policy | Mean history-pair density | Worst relative L2 |
|---|---:|---:|
| Legacy exact union | 25.00% | 0.8480 |
| Top-p 0.95 | 16.25% | 0.9042 |
| Top-p 0.90 | 13.22% | 0.9763 |
| Top-p 0.80 | 10.14% | 1.0324 |

All Top-p policies reduced scheduled history pairs without reducing transfer
bytes, and all increased capture error. Legacy exact union is the preliminary
quality-first winner; only legacy and Top-p 0.95 remain for the required
motion/state calibration. These errors are history-only because old captures do
not contain exact/current K/V, so they cannot replace complete-video review.

## Negative and recovery evidence

- The first query-policy batch failed 4/4 because compact route indices stayed
  on CPU while the union map was CUDA. Its bundle is retained; the new-SHA
  recovery passed 4/4.
- The first H replay failed 8/8 because persistent staging referenced an
  undefined local dtype. Its bundle is retained; the new-SHA recovery passed
  8/8.

## Next gates

1. Complete and review the two-state/two-seed RAG-Dense screen, then freeze the
   formal prompt manifest.
2. Run 39/120/240-latent end-to-end timelines to measure exposed CPU pack, H2D,
   Attention, non-Attention and VAE time.
3. Evaluate five static online utility candidates; do not run marginal-cost
   admission under the failed cost model.
4. Promote a transfer layout only if the same RoutePlan reduces exposed
   route+gather+H2D by at least 10% or meets the bandwidth gate without another
   component regression.
