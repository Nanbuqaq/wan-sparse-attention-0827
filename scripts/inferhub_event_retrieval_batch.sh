#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}" "${INFER_WEIGHTS_DIR:?}" "${INFER_OUTPUT_DIR:?}" "${VIRTUAL_ENV:?}"
event_root=$INFER_OUTPUT_DIR
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd "$INFER_CODE_DIR"
if [[ ${1:-} == --prepare-only ]]; then
  event_libraries=$VIRTUAL_ENV/lib
  for event_library in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib "$VIRTUAL_ENV"/lib/python*/site-packages/torch/lib "$VIRTUAL_ENV"/lib/python*/site-packages/triton/backends/nvidia/lib; do
    [[ ! -d "$event_library" ]] || event_libraries="$event_libraries:$event_library"
  done
  export LD_LIBRARY_PATH="$event_libraries:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="$INFER_WEIGHTS_DIR/python-overlay:$INFER_CODE_DIR"
  python -c 'import json,torch,av; from adapters.longlive_sparse.event_contrast_retrieval import EventContrastRetrieval; p=json.load(open("configs/system/event_retrieval_negative_controls.json")); assert p["design"]["total_cases"]==10; print("EVENT_RETRIEVAL_CPU_PREP_PASS",torch.__version__)'
  touch "$event_root/prepared.ok"
  exit 0
fi
source scripts/inferhub_runtime_env.sh
IFS=',' read -r -a event_gpus <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ ${#event_gpus[@]} == 4 ]] || { echo 'requires exactly four assigned GPUs' >&2; exit 2; }
event_seeds=(20260909 20260910 20260909 20260910)
event_pids=()
for event_lane in "${!event_gpus[@]}"; do
  event_extra=()
  if [[ $event_lane == 2 ]]; then event_extra+=(--novel-control duck); fi
  if [[ $event_lane == 3 ]]; then event_extra+=(--novel-control empty); fi
  CUDA_VISIBLE_DEVICES=${event_gpus[$event_lane]} python scripts/run_memory_revisit_videos.py \
    --output "$event_root/lane$event_lane" --scenario 1 --mode event_probe \
    --seed "${event_seeds[$event_lane]}" "${event_extra[@]}" >"$event_root/lane$event_lane.log" 2>&1 &
  event_pids+=("$!")
done
event_statuses=()
for event_pid in "${event_pids[@]}"; do
  event_status=0
  wait "$event_pid" || event_status=$?
  event_statuses+=("$event_status")
done
python scripts/collect_event_retrieval.py --root "$event_root" --exit-codes "${event_statuses[@]}"
