# Method-smoke completion audit (2026-08-28)

The stage intentionally reuses SHA-verified successes instead of regenerating
them. The 11 reused non-paper videos came from
`388728bc96f612c7935095276dac7a50a28fe29f`; the completion batch (including the
equal-length 100% pair) used
`d0983f43c27be6dafa9988f175c4d68457091633`; the decoupled local warm benchmark
used `c562a228dec0f1308ffc54fe5f8ed122c4a13f83`. The private full aggregation
audit has SHA-256
`ca19d5d8adff8b1339187e84c1a3fc625d9a76ec24f6505285747afa360b1c16`;
large videos, latents, captures and logs are intentionally excluded here.

- All 16 expected sparse-history method smokes reached verified pass states:
  81 decoded frames, finite `(1,21,16,60,104)` BF16 latents, declared backend,
  route-plan SHA and zero failed/fallback/NaN calls.
- RAG Dense39 reached 153 decoded frames with finite
  `(1,39,16,60,104)` BF16 latents and zero failed/fallback calls.
- Optimized Block64 at 100% reached 81 decoded frames with finite latents and
  zero failed/fallback calls. Its equal-length RAG Dense21 comparison was exact:
  `max_abs=0`, `relative_l2=0`, and identical BF16 latent values.
- Real-QKV calibration evaluated 52 candidates without formal prompts and froze
  SVG2 `q300/k512/i5`, AdaCluster `q65/k256/K7/Q11`, SVOO
  `q64/k256/i2/co1`, and SCOPE `q64/k333/top-p0.9`.

Warm real-shape backend medians (milliseconds) used one immutable route per row
and `5 warmup + 20 measured` iterations:

| History density | Grouped FA2 | Fixed64 | Varlen | Frozen-threshold result |
| ---: | ---: | ---: | ---: | --- |
| 0.10 | 3.1802 | 12.5016 | 12.8227 | grouped pass; fixed/varlen negative |
| 0.15 | 3.4074 | 15.4721 | 15.8565 | grouped pass; fixed/varlen negative |
| 0.25 | 3.2646 | 13.2783 | 13.6107 | grouped pass; fixed/varlen negative |
| 1.00 | 2.7717 | 3.3395 | 3.4171 | all pass |

At the three sparse densities, fixed/varlen replay had `max_abs=0.03125`,
relative L2 `0.000859--0.000966`, cosine above `0.9999995`, and the exact same
route-plan SHA as grouped FA2. The frozen different-kernel `max_abs=0.02`
threshold was not changed; those six records remain explicit kernel negatives.
