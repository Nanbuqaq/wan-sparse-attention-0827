# LongLive Sparse Attention

Training-free sparse-history routing and rectangular attention backends for
LongLive and LongLive-RAG. The branch compares cache/transfer-aware
autoregressive variants of Block, five distinct clustering directions, SVG2,
AdaCluster, SVOO and SCOPE.

## Evidence boundary

- `history_pair_density`, `history_transfer_density` and global executed
  density are separate metrics.
- Algorithms emit a backend-independent `HistoryRoutePlan`; kernel comparisons
  replay the same plan SHA.
- Native Dense has `routing_stage=N/A`; RAG Dense is post-transfer.
- Failed routing/kernel attempts remain explicit `fail` records and are not
  included in quality rankings.

## Setup

The repository uses public pinned submodules:

```bash
git submodule update --init --recursive
```

Model weights are not included. For the InferHub input bundle, run:

```bash
LONGLIVE_CONFIG_PATH=configs/inferhub/rag_dense_21.yaml \
  bash scripts/inferhub_entry.sh
```

CPU routing tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q -p no:cacheprovider tests
```

See `SOURCE_LOCK.json` and `THIRD_PARTY_NOTICES.md` for upstream commits,
licenses and modification boundaries.

