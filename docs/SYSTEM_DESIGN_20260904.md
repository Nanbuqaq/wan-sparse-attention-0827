# LongLive causal history-KV system design

## Boundary

The system separates four decisions:

```text
causal online evidence
  -> HistoryRoutePlan (logical Attention edges)
  -> TransferPlan (physical runs/pages/cache/copy)
  -> SystemCostModel (frozen prediction for a dataflow/hardware profile)
  -> measured SystemTraceRecord (post-run evidence only)
```

`OnlineRoutingContext` cannot contain full candidate K/V, Dense output,
same-call Dense Attention weights, or future video masks.  Those values are
restricted to `OfflineTeacherContext` in isolated analysis.

`TransferPlan` now records three byte levels separately: logical payload,
layout-expanded runs, and rectangular per-head padding.  A derived
`TransferExecutionPlan` chooses direct multi-run, packed-separate, or
packed-fused copying and owns the actual H2D copy count.  This keeps run
coalescing, CPU packing, and copy launch costs distinct in `SystemCostModel`.
Cost-aware admission remains disabled unless an independently held-out replay
set reaches MAPE at or below 15%.

## Physical block

An algorithm block is `(layer, head, global frame, within-frame Block64)`.
Page256 and Frame1560 are physical transfer layouts only.  Padding may increase
copied bytes but never adds logical Attention edges.

## Initial experiment order

1. Freeze a valid irreversible-state holdout using RAG Dense only.
2. Profile service time and critical-path exposed wait separately.
3. Coalesce the existing shared union without changing its route SHA.
4. Validate per-chunk denoising reuse and strict cache invalidation.
5. Add prediction-plus-validation-plus-completion prefetch.
6. Attempt KV-stationary execution only if GPU HBM/Attention remains exposed.

The short-video SVG2 audit showed that uniform exact-pair calibration can
destroy per-query probability mass.  LongLive therefore compares exact group
membership with mass-preserving Top-p inside the same transfer-bounded union;
logical pairs, scheduled/padded pairs and unique transferred KV remain separate
metrics.
