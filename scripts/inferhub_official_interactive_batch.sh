#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}" "${INFER_WEIGHTS_DIR:?}" "${INFER_OUTPUT_DIR:?}" "${VIRTUAL_ENV:?}"
interactive_root=$INFER_OUTPUT_DIR
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd "$INFER_CODE_DIR"
if [[ ${1:-} == --prepare-only ]]; then
  interactive_libs=$VIRTUAL_ENV/lib
  for interactive_lib in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib "$VIRTUAL_ENV"/lib/python*/site-packages/torch/lib "$VIRTUAL_ENV"/lib/python*/site-packages/triton/backends/nvidia/lib; do
    [[ ! -d "$interactive_lib" ]] || interactive_libs="$interactive_libs:$interactive_lib"
  done
  export LD_LIBRARY_PATH="$interactive_libs:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="$INFER_WEIGHTS_DIR/python-overlay:$INFER_CODE_DIR"
  python -c 'import torch,av; from adapters.longlive_sparse.stream_video_sink import IncrementalVideoSink; print("OFFICIAL_INTERACTIVE_PREP_PASS",torch.__version__)'
  touch "$interactive_root/prepared.ok";exit 0
fi
source scripts/inferhub_runtime_env.sh
IFS=',' read -r -a interactive_gpus <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ ${#interactive_gpus[@]} == 4 ]] || { echo 'requires exactly four assigned GPUs' >&2; exit 2; }
interactive_seeds=(20260909 20260910 20260909 20260910)
interactive_pids=()
for interactive_lane in "${!interactive_gpus[@]}"; do
  interactive_extra=()
  if [[ $interactive_lane == 2 ]]; then interactive_extra+=(--novel-control duck); fi
  if [[ $interactive_lane == 3 ]]; then interactive_extra+=(--novel-control empty); fi
  CUDA_VISIBLE_DEVICES=${interactive_gpus[$interactive_lane]} python scripts/run_official_interactive_local.py \
    --output "$interactive_root/lane$interactive_lane" --seed "${interactive_seeds[$interactive_lane]}" \
    "${interactive_extra[@]}" >"$interactive_root/lane$interactive_lane.log" 2>&1 &
  interactive_pids+=("$!")
done
interactive_statuses=()
for interactive_pid in "${interactive_pids[@]}"; do
  interactive_status=0
  wait "$interactive_pid" || interactive_status=$?
  interactive_statuses+=("$interactive_status")
done
python scripts/collect_official_interactive.py --root "$interactive_root" --exit-codes "${interactive_statuses[@]}"
