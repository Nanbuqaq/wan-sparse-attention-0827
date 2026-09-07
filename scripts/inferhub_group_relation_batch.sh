#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}" "${INFER_WEIGHTS_DIR:?}" "${INFER_OUTPUT_DIR:?}" "${VIRTUAL_ENV:?}"
group_length=${GROUP_RELATION_LATENT_FRAMES:-120}
group_root=$INFER_OUTPUT_DIR
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd "$INFER_CODE_DIR"
if [[ ${1:-} == --prepare-only ]]; then
  group_libraries=$VIRTUAL_ENV/lib
  for group_library in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib "$VIRTUAL_ENV"/lib/python*/site-packages/torch/lib "$VIRTUAL_ENV"/lib/python*/site-packages/triton/backends/nvidia/lib; do
    [[ ! -d "$group_library" ]] || group_libraries="$group_libraries:$group_library"
  done
  export LD_LIBRARY_PATH="$group_libraries:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="$INFER_WEIGHTS_DIR/python-overlay:$INFER_CODE_DIR"
  python -c 'import torch,av,flash_attn,triton; print("CPU_GROUP_PREP_PASS",torch.__version__,triton.__version__)'
  python scripts/build_group_relation_video_suite.py --latent-frames "$group_length" --output-dir "$group_root/control" --validate-only
  touch "$group_root/prepared.ok"
  exit 0
fi
source "$INFER_CODE_DIR/scripts/inferhub_runtime_env.sh"
IFS=',' read -r -a group_gpus <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ ${#group_gpus[@]} == 2 ]] || { echo 'requires exactly two assigned GPUs' >&2; exit 2; }
export LONGLIVE_NVTX=0 LONGLIVE_CAPTURE_QKV=0 LONGLIVE_CAPTURE_COMPLETE_ATTENTION=0
python scripts/build_group_relation_video_suite.py --latent-frames "$group_length" --output-dir "$group_root/control"
group_pids=()
for group_lane in "${!group_gpus[@]}"; do
  mkdir -p "$group_root/lane$group_lane"
  CUDA_VISIBLE_DEVICES=${group_gpus[$group_lane]} INFER_OUTPUT_DIR=$group_root/lane$group_lane \
    python scripts/run_loaded_method_suite.py --suite "$group_root/control/lane$group_lane.json" \
    --shard-index 0 --shard-count 1 >"$group_root/lane$group_lane/runner.log" 2>&1 &
  group_pids+=("$!")
done
group_statuses=()
for group_pid in "${group_pids[@]}"; do
  group_status=0
  wait "$group_pid" || group_status=$?
  group_statuses+=("$group_status")
done
python scripts/collect_hierarchical_pair_states.py --root "$group_root" --exit-codes "${group_statuses[@]}"
