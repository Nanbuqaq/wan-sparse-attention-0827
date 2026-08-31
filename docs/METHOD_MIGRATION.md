# LongLive autoregressive method migration contracts

All values in the method catalogue are either source-derived starting points or
explicit LongLive transfer settings. They are not described as formal LongLive
optima. Final values are frozen only after calibration on captured LongLive
Q/K tensors, without using formal evaluation prompts.

## Routing stages

| Method group | Stage | LongLive autoregressive interpretation |
| --- | --- | --- |
| Native Dense / Native Block | N/A | Native rolling GPU cache; no CPU history archive transfer |
| RAG Dense, Oracle, SVOO, SCOPE | post-transfer | Move the complete RAG frame-candidate KV set, then route on GPU |
| SVG2, AdaCluster, QMetric | hybrid | LongLive-RAG frame retrieval is the pre-transfer coarse stage; Q/K fine routing occurs on GPU |
| Block, Random, Local and pre-transfer clusters | pre-transfer | Route from CPU archive metadata/history, then move only the selected union |
| Coverage/V-aware/transfer-aware proposed routes | pre-transfer | Build Q summaries on GPU, route only against CPU K/V prototypes, then move selected original KV |

Every method emits a `HistoryRoutePlan` that defines per-query-group Q-K pairs,
the unique transferred KV union and the execution backend. Fixed, grouped and
varlen kernel comparisons replay this plan unchanged.

## Paper-method starting points

- SVG2: the pinned Sparse-VideoGen Wan 720p T2V launch script uses Q=300,
  K=1000, top-p=0.90, minimum K-cluster ratio 0.10, 50 initialization
  iterations and 2 update iterations. LongLive keeps Q=300/K=1000/top-p=0.90
  as the source-derived starting point but calibrates iteration/reuse settings
  for its rectangular current-Q/history-K shape.
- SVOO: the pinned Wan 1.3B script uses Q=256, K=1024, top-p=0.90, 2 initial
  and 2 update iterations plus reuse. The LongLive version applies co-clustered
  Q/K routing to the RAG candidate history and uses exact token budgets.
- AdaCluster: the pinned Wan source uses thresholded K/Q clustering with K
  distance threshold 5.5 and Q threshold 9.0. Because the upstream repository
  has no root license, this branch contains an independent implementation and
  copies no source.
- SCOPE: the autoregressive version clusters full current Q, builds three
  Wan-RoPE key-subspace proxies, applies top-p=0.90 with a 10% fixed floor as a
  routing prior, and then enforces the shared exact history-pair budget.

## Five self-designed clustering directions

- QLocal-KMeans8: local current-Q clusters route to history clusters.
- Radius-K256: cluster score plus a residual-radius upper-bound term.
- QMetric-K256-R32: a query-covariance low-rank metric before history routing.
- Temporal-K256-T16: temporally stratified history clusters.
- SizeSplit-K128-C2: capacity-aware recursive splitting of oversized clusters.

KCluster32 and Fixed-K128/256 remain simple-K baselines and do not count toward
the five directions. Random, Block, Local and Oracle remain baselines.

## Proposed paper-method ablations

- `coverage_cluster_history`: preserves a fixed Block/content tier and an
  explicit time/spatial coverage tier; only the remaining history budget is
  ranked by cluster prototypes.
- `vaware_cluster_history`: adds an online value proxy computed as a
  probability proxy times CPU V-prototype norm. It never observes dense output
  or complete candidate KV.
- `transfer_vaware_hybrid_history`: first builds the same per-query route, then
  restricts all groups to a shared history union whose size is controlled
  separately from history-pair density.

The 70/15/15 and 80/10/10 splits are initial candidates, not formal settings.
Exact dense-versus-sparse output residual is used only by the isolated offline
teacher. Parameters remain unfrozen until captured-QKV ranking and two
non-formal 477-frame calibration prompts both pass.
