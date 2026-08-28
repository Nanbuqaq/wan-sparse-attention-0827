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

## Stage runners

- `scripts/inferhub_batch_gate_v3.sh`: isolated GPU routing and backend gates,
  Block64 100% and RAG cache-roll/QKV capture.
- `scripts/inferhub_batch_calibrate_methods.sh`: corrected RAG/Block replay,
  real-QKV paper calibration, all-method 21-frame smokes and warmed identical-
  route backend benchmarking.
- `scripts/inferhub_prepare_method_calibration.sh` and
  `scripts/inferhub_batch_complete_methods_v2.sh`: CPU-prep real-QKV
  calibration followed by four disjoint GPU lanes for matched 100% correctness,
  four paper-method smokes and warm backend benchmarking.
- `scripts/inferhub_batch_dense_screen.sh`: Dense-only eight-prompt, two-seed
  screening with Native Dense and RAG Dense on different GPUs.
- `scripts/build_dense_review_table.py` and `scripts/freeze_dense_prompts.py`:
  freeze formal prompts without consuming any sparse result.
- `scripts/build_formal_suites.py` and `scripts/inferhub_batch_basic_477.sh`:
  generate and run two 477-frame base cases for every method.
- `scripts/evaluate_videos.py`, `scripts/summarize_video_cases.py` and
  `scripts/analyze_complexity.py`: paired fidelity, complete-video bootstrap,
  Pareto expansion and separated theoretical/measured complexity.

Load-once runners write per-case terminal states and can reuse only successful
cases whose video SHA, full decoded-frame count and latent artifact all verify.

See `SOURCE_LOCK.json` and `THIRD_PARTY_NOTICES.md` for upstream commits,
licenses and modification boundaries.
