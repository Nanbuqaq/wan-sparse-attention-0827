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
