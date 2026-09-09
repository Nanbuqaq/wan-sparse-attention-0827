# Native VAE memory-format boundary

The completed gate is under
`results/metrics/memory_activation_20260910/native_vae_layout_gate16_v2/`.
Runtime fda408c; four modes on a saved 16-latent native-resolution prefix.

Baseline streaming decode is float/RGB exact. Three channels-last layout
variants preserve parameter values but change float/RGB output (maximum0.09375,
relative L2~0.0020 against native BF16, not FP32). All three share the same RGB
hash on this prefix. None enters the lossless pipeline; no speedup is inferred
from diagnostic wall time, which includes CPU checking.

Peak allocated observation is12.093→11.151GB; reserved stays15.234GB. Full-conv
input hooks introduce49 reformats/~8.368GB logical payload. Neither figure is
an HBM counter or an E2E result. A possible approximate-layout study must retain
this numerical drift and cannot borrow the earlier exact-pipeline qualification.

The previous v1 failure was an audit fingerprint issue with singleton strides,
fixed using canonical 1D-owned bytes and CPU regression tests. Its artifacts
remain unchanged. This is not evidence that the layout computation itself failed.

## Independent synchronized timing completed

Runtime667e44c,5warmups and30randomized paired decoder-only measurements.
Both modes reproduce their own gate RGB hashes. Baseline median9.575923s/p95
9.576900s; weights-only9.257098s/p959.258167s. Latency reduction~3.33%, below the
registered10% follow-up criterion; stop this branch without online promotion.
Some CPU tests overlapped this diagnostic; do not claim isolated production
latency. Raw pair order/timings and numerical drift remain in
`results/metrics/memory_activation_20260910/native_vae_layout_timing16_v1/`.
