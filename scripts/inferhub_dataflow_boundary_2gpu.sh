#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}"
: "${INFER_WEIGHTS_DIR:?}"
: "${INFER_OUTPUT_DIR:?}"
: "${CUDA_VISIBLE_DEVICES:?}"
: "${VIRTUAL_ENV:?}"
export LONGLIVE_INPUT_BUNDLE_ROOT="$INFER_WEIGHTS_DIR/input_bundle"
source "$INFER_CODE_DIR/scripts/inferhub_runtime_env.sh"
if [[ ${LONGLIVE_PRIVATE_TRITON331:-0} == 1 ]]; then
  [[ -f $INFER_OUTPUT_DIR/private-triton331/ready.json ]] || exit 2
  export PYTHONPATH="$INFER_OUTPUT_DIR/private-triton331:$PYTHONPATH"
fi
IFS=',' read -r -a assigned_gpus <<<"$CUDA_VISIBLE_DEVICES"
[[ ${#assigned_gpus[@]} == 2 ]] || exit 2
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
cd "$INFER_CODE_DIR"
python scripts/record_dataflow_hardware.py --output "$INFER_OUTPUT_DIR/hardware.json"
points=$(python -c 'import json;print(",".join(map(str,json.load(open("configs/system/dataflow_boundary_points.json"))["case_indices"])))')
triton_boundary_cache=$(mktemp -d /tmp/longlive-boundary-triton.XXXXXX)
pids=()
for lane in 0 1; do
  gate_args=()
  [[ $lane == 0 ]] || gate_args+=(--large)
  CUDA_VISIBLE_DEVICES=${assigned_gpus[$lane]} TRITON_CACHE_DIR=$triton_boundary_cache/lane$lane \
    python scripts/gate_dataflow_references.py --streaming "${gate_args[@]}" \
      --output "$INFER_OUTPUT_DIR/hardware_gate$lane.json" >"$INFER_OUTPUT_DIR/hardware_gate$lane.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ $failed != 0 ]]; then
  echo 'hardware numerical gate failed; boundary matrix not started' >&2
  exit 2
fi
pids=()
for lane in 0 1; do
  CUDA_VISIBLE_DEVICES=${assigned_gpus[$lane]} TRITON_CACHE_DIR=$triton_boundary_cache/lane$lane \
    python scripts/benchmark_dataflow_matrix.py --case-indices "$points" --lane "$lane" --lanes 2 \
      --warmup 5 --repeats 30 --output "$INFER_OUTPUT_DIR/lane$lane" >"$INFER_OUTPUT_DIR/lane$lane.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
