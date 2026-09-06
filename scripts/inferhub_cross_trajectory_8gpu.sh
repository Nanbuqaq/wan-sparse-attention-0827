#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}"
: "${INFER_WEIGHTS_DIR:?}"
: "${INFER_OUTPUT_DIR:?}"
: "${CUDA_VISIBLE_DEVICES:?}"
IFS=',' read -r -a assigned_gpus <<<"${CUDA_VISIBLE_DEVICES}"
[[ ${#assigned_gpus[@]} == 8 ]] || { echo 'requires8 assigned GPUs' >&2; exit 2; }
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NVIDIA_TF32_OVERRIDE=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd "$INFER_CODE_DIR"
pids=()
for lane in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=${assigned_gpus[$lane]} python scripts/run_cross_trajectory_lane.py \
    --input-root "$INFER_WEIGHTS_DIR" --lane "$lane" --lanes 8 \
    --output "$INFER_OUTPUT_DIR/lane$lane" >"$INFER_OUTPUT_DIR/lane$lane.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
