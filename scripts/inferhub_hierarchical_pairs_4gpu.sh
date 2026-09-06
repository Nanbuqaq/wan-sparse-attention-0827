#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}"
: "${INFER_WEIGHTS_DIR:?}"
: "${INFER_OUTPUT_DIR:?}"
: "${CUDA_VISIBLE_DEVICES:?}"
: "${VIRTUAL_ENV:?}"
export LONGLIVE_INPUT_BUNDLE_ROOT="$INFER_WEIGHTS_DIR/input_bundle"
source "$INFER_CODE_DIR/scripts/inferhub_runtime_env.sh"
IFS=',' read -r -a assigned_gpus <<<"$CUDA_VISIBLE_DEVICES"
[[ ${#assigned_gpus[@]} == 4 ]] || exit 2
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LONGLIVE_CAPTURE_COMPLETE_ATTENTION=0 LONGLIVE_NVTX=0
cd "$INFER_CODE_DIR"
batch_root=$INFER_OUTPUT_DIR
python scripts/build_hierarchical_video_pairs.py --latent-frames 120 --output-dir "$batch_root/control"
pids=()
for lane in 0 1 2 3; do
  mkdir -p "$batch_root/lane$lane"
  CUDA_VISIBLE_DEVICES=${assigned_gpus[$lane]} INFER_OUTPUT_DIR=$batch_root/lane$lane \
    python scripts/run_loaded_method_suite.py --suite "$batch_root/control/lane$lane.json" \
    --shard-axis case --shard-index 0 --shard-count 1 >"$batch_root/lane$lane/runner.log" 2>&1 &
  pids+=("$!")
done
statuses=()
for pid in "${pids[@]}"; do
  code=0; wait "$pid" || code=$?
  statuses+=("$code")
done
python scripts/collect_hierarchical_pair_states.py --root "$batch_root" --exit-codes "${statuses[@]}"
