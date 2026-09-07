#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}" "${INFER_WEIGHTS_DIR:?}" "${INFER_OUTPUT_DIR:?}" "${VIRTUAL_ENV:?}"
prototype_root=$INFER_OUTPUT_DIR
prototype_libraries=$VIRTUAL_ENV/lib
for prototype_library in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib "$VIRTUAL_ENV"/lib/python*/site-packages/torch/lib "$VIRTUAL_ENV"/lib/python*/site-packages/triton/backends/nvidia/lib; do
  [[ ! -d "$prototype_library" ]] || prototype_libraries="$prototype_libraries:$prototype_library"
done
export LD_LIBRARY_PATH="$prototype_libraries:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$INFER_CODE_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd "$INFER_CODE_DIR"
if [[ ${1:-} == --prepare-only ]]; then
  python -c 'import torch,numpy; print("PROTOTYPE_CPU_ENV_PASS",torch.__version__)'
  python scripts/stage_sprint_capture_inputs.py --verify-manifest "$INFER_WEIGHTS_DIR/manifest.json"
  touch "$prototype_root/prepared.ok"
  exit 0
fi
IFS=',' read -r -a prototype_gpus <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ ${#prototype_gpus[@]} == 4 ]] || { echo 'requires exactly four assigned GPUs' >&2; exit 2; }
prototype_kinds=(motion motion state state)
prototype_layers=(0,9 19,29 0,9 19,29)
prototype_pids=()
for prototype_lane in "${!prototype_gpus[@]}"; do
  mkdir -p "$prototype_root/lane$prototype_lane"
  CUDA_VISIBLE_DEVICES=${prototype_gpus[$prototype_lane]} \
    python scripts/probe_prototype_tail.py --capture-manifest "$INFER_WEIGHTS_DIR/manifest.json" \
    --kind "${prototype_kinds[$prototype_lane]}" --layers "${prototype_layers[$prototype_lane]}" \
    --residual-controls --round-prototypes-bf16 --output "$prototype_root/lane$prototype_lane/results" \
    >"$prototype_root/lane$prototype_lane/runner.log" 2>&1 &
  prototype_pids+=("$!")
done
prototype_statuses=()
for prototype_pid in "${prototype_pids[@]}"; do
  prototype_status=0
  wait "$prototype_pid" || prototype_status=$?
  prototype_statuses+=("$prototype_status")
done
python scripts/collect_prototype_replay.py --root "$prototype_root" --exit-codes "${prototype_statuses[@]}"
