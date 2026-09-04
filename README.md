# Wan Sparse Attention 0827 — Short Video

This branch contains the Wan2.1 short-video sparse-attention implementation and
reproducible experiment harness. It separates routing algorithms from execution
backends, records actual Q-K pair density, and disables silent Dense fallbacks.

## Method groups

- Baselines: Dense, Original Block, Random Block, 3D Local Window, Fixed-K,
  QSort-Local8, and Token Oracle.
- Paper adapters: SVG2, AdaCluster, SVOO, and SCOPE.
- Clean-room clustering families: capacity-balanced, radius-adaptive,
  hierarchical, product-quantized, spatiotemporal-constrained, and
  query-metric/PCA clustering.

Execution backends are independent of routing:

- `fixed64_bf16`: clean-room Triton fixed-64 block-sparse attention;
- `varlen_triton_native`: pinned SVOO variable-block implementation;
- `varlen_triton_csr`: CSR-indexed clean-room variable-block implementation.

Stage-3 adds three original-order routes without changing the frozen Stage-2
evidence: `coverage_cluster` reserves the exact budget for Block/local coverage
before remote cluster retrieval, `vaware_cluster` ranks only that remote
remainder with query-conditioned V information, and `stage3_hybrid` adds a
low-frequency refresh and denoise-phase schedule while keeping the same total
Q-K pair density.

## Environment

Set a local Diffusers model path before running commands:

```bash
export WAN_MODEL_PATH=/path/to/Wan2.1-T2V-1.3B-Diffusers
```

Optional Python overlay directories can be supplied through the
`WAN_SPARSE_PYTHON_OVERLAYS` path list. Runtime caches are confined to
`.runtime/cache`.

## Validation and execution

```bash
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 scripts/validate_correctness_v2.py --include-full-shapes
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 scripts/check_latent_equivalence.py
/usr/bin/python3 scripts/run_on_free_gpu.py -- \
  /usr/bin/python3 scripts/run_matrix.py --suite configs/calibration_50step_v2.json
```

All Stage-3 GPU commands use the workflow-wide lock and one shard:

```bash
/usr/bin/python3 scripts/run_on_free_gpu.py \
  --global-lock /tmp/wan_short_stage3_single_gpu.lock \
  --wait-for-global-lock -- \
  /usr/bin/python3 scripts/run_matrix.py \
  --suite configs/stage3_smoke_4step.json \
  --num-shards=1 --shard-index=0
```

`scripts/stage3_preflight.py` records CUDA availability and verifies that every
Stage-3 suite has an isolated output/manifest root and the required lock policy.
The 50-step formal template must not be run until captured-QKV screening and the
isolated 50-step calibration have frozen its parameters.

## Stage-3 result

The frozen Stage-3 suite completed with 49/49 tasks and a 22/22 final audit.
At 25% actual Q-K pair density, the final hybrid preserves normal visual output
for all four main prompts, a second seed, and two negative cases. It averages
12.690 dB PSNR to matched Dense at 1.208x end-to-end speed, versus 12.320 dB at
1.340x for Original Block. The four main cases are all PSNR wins over Block,
but the complete-case Holm-adjusted test remains non-significant because the
sample contains only four videos.

See `docs/FINAL_REPORT_STAGE3.md` for the V-aware, 100% numerical, CSR, visual
review, and LongLive migration conclusions.

The 2026-09-04 SVG2 debug separates the upstream Top-p policy from the frozen
exact-25% adaptation. On a unified-varlen 2x2, both exact-budget cells collapse
and both Top-p cells preserve the subject; see
`docs/SVG2_DEBUG_2026-09-04.md`. This is a Wan-14B/720p policy transfer to
Wan-1.3B/480p, not a complete paper reproduction or a LongLive quality result.

The runner writes one MP4 and one task-scoped stats JSON per completed task.
Suite-v2 evaluation reads only tasks expanded from the frozen suite; it never
recursively scans an output root.

## Evidence policy

- Four-step runs are correctness/OOM/density smoke tests only.
- Parameter freezing and quality conclusions use complete 50-step videos.
- Statistical samples are complete prompt/seed video cases, never individual
  frames.
- Failed, OOM, negative, reused, and new results remain separately labelled.
- Model weights, videos, full metrics, caches, logs, and internal data are not
  part of the public repository.

See `docs/LICENSE_AUDIT.md` and `docs/UPSTREAM_LOCK.md` for provenance and
redistribution constraints.
