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
