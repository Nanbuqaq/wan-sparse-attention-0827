#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}" "${INFER_OUTPUT_DIR:?}" "${INFER_WEIGHTS_DIR:?}" "${CUDA_VISIBLE_DEVICES:?}"
IFS=, read -r -a tether_gpus <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#tether_gpus[@]} -eq 4 ]] || exit 2
tether_pids=()
for tether_lane in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=${tether_gpus[$tether_lane]} python scripts/run_official_tether_batch.py --lane "$tether_lane" --lanes 4 \
    >"$INFER_OUTPUT_DIR/launcher_lane$tether_lane.log" 2>&1 &
  tether_pids+=("$!")
done
tether_status=0
for tether_pid in "${tether_pids[@]}"; do
  wait "$tether_pid" || tether_status=1
done
exit "$tether_status"
