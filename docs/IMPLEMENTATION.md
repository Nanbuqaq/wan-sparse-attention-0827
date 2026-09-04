# Stage-2 implementation and reproduction

## Architecture

- `adapters/routes/` separates baseline, paper-derived, and six clean-room clustering families.
- `adapters/routing.py` owns exact Q-K pair budgeting, permutation, RoutePlan materialization, padding, and load accounting.
- `adapters/kernels_fixed64.py` is an independently written fixed-64 BF16 Triton backend.
- `adapters/kernels_varlen_csr.py` stores active K-cluster columns in CSR form and schedules one program per Q tile.
- `adapters/dependencies.py` builds task-scoped dependency manifests using symbol-level AST hashes and exact runtime/source files.
- `adapters/wan_sparse.py` only replaces Wan self-attention and raises on failure; no Dense fallback is present.

## Evidence sequence

1. Run route and backend-shape correctness, including full Wan sequence length.
2. Capture real Q/K/V and retain two parameter candidates per paper/self method.
3. Freeze parameters from isolated 50-step calibration videos.
4. Freeze formal prompts using Dense-only 50-step visual review.
5. Build and run the immutable formal suite; do not retune after seeing it.

```bash
export WAN_MODEL_PATH=/path/to/Wan2.1-T2V-1.3B-Diffusers
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 scripts/validate_correctness_v2.py --include-full-shapes
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 scripts/capture_qkv_v2.py
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 scripts/screen_captured_qkv.py --capture-manifest <manifest>
/usr/bin/python3 scripts/build_calibration_suite_v2.py
/usr/bin/python3 scripts/run_on_free_gpu.py -- /usr/bin/python3 scripts/run_matrix.py --suite configs/calibration_50step_v2.json
```

Suite-v2 evaluation expands the frozen suite and reads only the expected stats paths. A task can reuse an older artifact only when its task-scoped generation, route, processor, backend, runtime, model, and configuration dependencies match.

Timing separates cold kernel calls, warm kernel calls, CSR planning, clustering, permutation, selection, inverse permutation, generation, export, and peak allocated/reserved memory. Quality statistics use complete video cases rather than individual frames.

## Stage-3 stability and V-aware routes

Stage-3 is isolated from the completed Stage-2 suite. The new routes never
replace K/V with centroids and never reorder the executed token stream:

- `coverage_cluster` protects a configurable fraction of Original-Block edges,
  then adds explicit local/time edges, and spends only the remainder on remote
  cluster retrieval.
- `vaware_cluster` keeps the same protected edges and same final pair budget;
  query-conditioned V-prototype or output-residual scores may reorder only the
  remote remainder.
- `stage3_hybrid` uses the residual objective, low-frequency cluster refresh,
  and a denoise-phase allocation schedule. Its total density is unchanged on
  every Attention call, not merely on average.

The fixed and CSR backends receive the same RoutePlan. A CSR failure only
blocks the CSR performance claim; the fixed backend remains available for video
quality coverage. Four-step suites are smoke evidence only. Parameter selection
and all conclusions remain gated on isolated 50-step calibration and normal
50-step formal videos.

## SVG2 Dense-guard control

`MethodConfig.svg2_dense_guard` is tri-state. `None` preserves historical
behavior (`svg2_official_top_p` guarded, exact-density SVG2 routes unguarded),
`True` enables the same step/layer guard for an exact-density ablation, and
`False` disables it for a Top-p ablation. Stats record the floor-rounded
`dense_steps`, `dense_layers`, expected call counts, actual call counts, and
whether they match. The guard is an explicit Dense reference path, never a
fallback.
