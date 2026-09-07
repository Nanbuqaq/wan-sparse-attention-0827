#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}" "${INFER_WEIGHTS_DIR:?}" "${INFER_OUTPUT_DIR:?}" "${VIRTUAL_ENV:?}"
revisit_root=$INFER_OUTPUT_DIR
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd "$INFER_CODE_DIR"
if [[ ${1:-} == --prepare-only ]]; then
  revisit_libraries=$VIRTUAL_ENV/lib
  for revisit_library in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib "$VIRTUAL_ENV"/lib/python*/site-packages/torch/lib "$VIRTUAL_ENV"/lib/python*/site-packages/triton/backends/nvidia/lib; do
    [[ ! -d "$revisit_library" ]] || revisit_libraries="$revisit_libraries:$revisit_library"
  done
  export LD_LIBRARY_PATH="$revisit_libraries:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="$INFER_WEIGHTS_DIR/python-overlay:$INFER_CODE_DIR"
  python -c 'import json,torch,av; from adapters.longlive_sparse.scheduled_prompts import validate_schedule; p=json.load(open("configs/system/memory_revisit_development.json")); assert len(p["scenarios"])*len(p["seeds"])==4; [validate_schedule(s["segments"],latent_frames=120) for s in p["scenarios"]]; print("REVISIT_DENSE_CPU_PREP_PASS",torch.__version__)'
  touch "$revisit_root/prepared.ok"
  exit 0
fi
source scripts/inferhub_runtime_env.sh
IFS=',' read -r -a revisit_gpus <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ ${#revisit_gpus[@]} == 4 ]] || { echo 'requires exactly four assigned GPUs' >&2; exit 2; }
revisit_scenarios=(0 0 1 1)
revisit_seeds=(20260909 20260910 20260909 20260910)
revisit_pids=()
for revisit_lane in "${!revisit_gpus[@]}"; do
  CUDA_VISIBLE_DEVICES=${revisit_gpus[$revisit_lane]} python scripts/run_memory_revisit_videos.py \
    --output "$revisit_root/lane$revisit_lane" --scenario "${revisit_scenarios[$revisit_lane]}" \
    --seed "${revisit_seeds[$revisit_lane]}" --mode dense_screen >"$revisit_root/lane$revisit_lane.log" 2>&1 &
  revisit_pids+=("$!")
done
revisit_statuses=()
for revisit_pid in "${revisit_pids[@]}"; do
  revisit_status=0
  wait "$revisit_pid" || revisit_status=$?
  revisit_statuses+=("$revisit_status")
done
python scripts/collect_revisit_dense_screen.py --root "$revisit_root" --exit-codes "${revisit_statuses[@]}"
