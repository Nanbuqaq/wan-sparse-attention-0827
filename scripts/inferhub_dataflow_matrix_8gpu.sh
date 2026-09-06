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
[[ ${#assigned_gpus[@]} == 8 ]] || exit 2
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
cd "$INFER_CODE_DIR"
batch_root=$INFER_OUTPUT_DIR
pids=()
for lane in "${!assigned_gpus[@]}"; do
  CUDA_VISIBLE_DEVICES=${assigned_gpus[$lane]} python scripts/benchmark_dataflow_matrix.py \
    --lane "$lane" --lanes 8 --warmup 5 --repeats 30 --output "$batch_root/lane$lane" \
    >"$batch_root/lane$lane.log" 2>&1 &
  pids+=("$!")
done
statuses=()
for pid in "${pids[@]}"; do
  code=0; wait "$pid" || code=$?
  statuses+=("$code")
done
python scripts/collect_dataflow_matrix.py --root "$batch_root" --exit-codes "${statuses[@]}"
# Representative CUDA activity only; not a full matrix/video profile.
if command -v nsys >/dev/null 2>&1; then
  mkdir -p "$batch_root/profile"
  CUDA_VISIBLE_DEVICES=${assigned_gpus[0]} nsys profile --trace=cuda,nvtx --sample=none \
    --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop \
    --output="$batch_root/profile/kv_case71" \
    python scripts/benchmark_dataflow_matrix.py --case-index 71 --warmup 1 --repeats 1 \
      --profile-scope streaming_overlap --profile-backend kvout --output "$batch_root/profile/replay" \
      >"$batch_root/profile/runner.log" 2>&1 || true
fi
python -c 'import json,sys;from pathlib import Path;d=json.loads((Path(sys.argv[1])/"terminal.json").read_text());sys.exit(0 if d["status"]=="pass" else 1)' "$batch_root"
