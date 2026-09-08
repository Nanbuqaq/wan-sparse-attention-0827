#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}" "${INFER_WEIGHTS_DIR:?}" "${INFER_OUTPUT_DIR:?}" "${VIRTUAL_ENV:?}"
native_root=$INFER_OUTPUT_DIR
native_libs=$VIRTUAL_ENV/lib
for native_lib in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib "$VIRTUAL_ENV"/lib/python*/site-packages/torch/lib "$VIRTUAL_ENV"/lib/python*/site-packages/triton/backends/nvidia/lib; do
  [[ ! -d "$native_lib" ]] || native_libs="$native_libs:$native_lib"
done
export LD_LIBRARY_PATH="$native_libs:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$INFER_WEIGHTS_DIR/python-overlay:$INFER_CODE_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LLV2_USE_FA3=0 LLV2_USE_FA4=0 LLV2_USE_TE_ATTN=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$INFER_CODE_DIR"
if [[ ${1:-} == --prepare-only ]]; then
  python scripts/prepare_longlive2_reference.py --inputs "$INFER_WEIGHTS_DIR" --source "$INFER_CODE_DIR/third_party/LongLive2"
  touch "$native_root/prepared.ok";exit 0
fi
IFS=',' read -r -a native_gpus <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ ${#native_gpus[@]} == 4 ]] || { echo 'requires four assigned GPUs' >&2; exit 2; }
native_seeds=(20260909 20260910 20260909 20260910)
native_pids=()
for native_lane in "${!native_gpus[@]}"; do
  native_extra=()
  if [[ $native_lane == 2 ]]; then native_extra+=(--control duck); fi
  if [[ $native_lane == 3 ]]; then native_extra+=(--control empty); fi
  CUDA_VISIBLE_DEVICES=${native_gpus[$native_lane]} python scripts/run_longlive2_native_reference.py \
    --assets "$INFER_WEIGHTS_DIR" --output "$native_root/lane$native_lane" --seed "${native_seeds[$native_lane]}" \
    "${native_extra[@]}" >"$native_root/lane$native_lane.log" 2>&1 &
  native_pids+=("$!")
done
native_statuses=()
for native_pid in "${native_pids[@]}"; do
  native_status=0
  wait "$native_pid" || native_status=$?
  native_statuses+=("$native_status")
done
python scripts/collect_longlive2_reference.py --root "$native_root" --exit-codes "${native_statuses[@]}"
