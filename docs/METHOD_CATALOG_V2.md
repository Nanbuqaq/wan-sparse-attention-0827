# Stage-2 method catalog

## Baselines

| ID | Category | Definition |
|---|---|---|
| `dense` | quality reference | Full Wan self-attention. |
| `block` | fixed-block baseline | Mean Q/K descriptors over contiguous 64-token blocks with an exact global edge budget. |
| `random` | random baseline | Deterministic random fixed-block graph with the same executed edge budget. |
| `local_3d` | local baseline | Nearest fixed blocks under normalized latent T/H/W distance. |
| `fixed_k128` | clustering baseline | One flat post-RoPE K partition with K=128; other K values are ablations, not additional method families. |
| `qsort_local8` | layout baseline | Local Q clustering used only to reorder each 64-token Q block before block retrieval. It is not counted as a self-clustering family. |
| `token_oracle` | non-deployable baseline | Dense QK scores choose a fixed token budget before fixed-block execution. |

## Paper methods

| ID | Provenance | Core migrated behavior |
|---|---|---|
| `svg2` | Sparse-VideoGen Apache-2.0 | Independent Euclidean Q/K clustering, size-weighted centroid scores, semantic permutation, exact executed-pair calibration. |
| `adacluster` | clean-room; upstream has no repository license | Normalized Q clustering, residual-triggered adaptive K clusters, sign-aware cluster upper bound, low-frequency refresh. |
| `svoo` | SVOO Apache-2.0 | Bidirectional Q/K co-clustering, size-weighted cluster graph, configurable refresh/reuse. |
| `scope` | paper-derived; no official code claimed | Full-Q clustering and T/H/W RoPE-subspace K lookup, mapped to original-token fixed execution. |

All Q/K counts, thresholds, iterations, and refresh policies begin as calibration candidates. They are frozen only after captured-QKV screening and isolated 50-step video comparison.

## Six clean-room clustering families

| ID | Distinct clustering direction |
|---|---|
| `capacity_balanced` | Flat K-means followed by capacity-constrained splitting of oversized clusters. |
| `radius_adaptive` | DP-means-style residual/outlier seeds add clusters beyond a coarse initial partition. |
| `hierarchical` | Coarse partition followed by branch-initialized fine clusters. |
| `product_quantized` | Independent feature-subspace codebooks; observed product codes define token groups. |
| `spatiotemporal` | K-means on content features augmented by normalized latent T/H/W coordinates. |
| `query_metric` | K is clustered in a low-rank metric induced by the current Q covariance eigenspace. |

These six entries are counted separately because they use different clustering objectives or representations. QSort and Fixed-K parameter changes are excluded from this count.

## Backend axis

The route family and execution backend are independent:

- `fixed64_bf16`: clean-room fixed 64×64 block-sparse Triton kernel;
- `varlen_triton_native`: pinned SVOO dynamic-block Triton kernel;
- `varlen_triton_csr`: active-column CSR plan with independent Q-tile scheduling.

Cross-backend speed comparisons reuse the same RoutePlan and graph hash. A CSR failure does not remove the route family from fixed/native quality experiments.
