#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}" "${INFER_WEIGHTS_DIR:?}" "${INFER_OUTPUT_DIR:?}" "${VIRTUAL_ENV:?}"
episode_root=$INFER_OUTPUT_DIR
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd "$INFER_CODE_DIR"
if [[ ${1:-} == --prepare-only ]]; then
  episode_libs=$VIRTUAL_ENV/lib
  for episode_lib in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib "$VIRTUAL_ENV"/lib/python*/site-packages/torch/lib "$VIRTUAL_ENV"/lib/python*/site-packages/triton/backends/nvidia/lib; do
    [[ ! -d "$episode_lib" ]] || episode_libs="$episode_libs:$episode_lib"
  done
  export LD_LIBRARY_PATH="$episode_libs:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="$INFER_WEIGHTS_DIR/python-overlay:$INFER_CODE_DIR"
  python -c 'import torch,av; from adapters.longlive_sparse.episodic_snapshot_probe import EpisodicSnapshotProbe; print("EPISODE_SNAPSHOT_PREP_PASS",torch.__version__)'
  touch "$episode_root/prepared.ok";exit 0
fi
source scripts/inferhub_runtime_env.sh
IFS=',' read -r -a episode_gpus <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ ${#episode_gpus[@]} == 2 ]] || { echo 'requires exactly two assigned GPUs' >&2; exit 2; }
episode_seeds=(20260909 20260910)
episode_pids=()
for episode_lane in "${!episode_gpus[@]}"; do
  CUDA_VISIBLE_DEVICES=${episode_gpus[$episode_lane]} python scripts/run_official_interactive_local.py \
    --snapshot-probe --output "$episode_root/lane$episode_lane" --seed "${episode_seeds[$episode_lane]}" \
    >"$episode_root/lane$episode_lane.log" 2>&1 &
  episode_pids+=("$!")
done
episode_statuses=()
for episode_pid in "${episode_pids[@]}"; do
  episode_status=0
  wait "$episode_pid" || episode_status=$?
  episode_statuses+=("$episode_status")
done
python scripts/collect_episode_snapshot.py --root "$episode_root" --exit-codes "${episode_statuses[@]}"
