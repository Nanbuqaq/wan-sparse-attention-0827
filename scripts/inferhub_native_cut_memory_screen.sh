#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}" "${INFER_WEIGHTS_DIR:?}" "${INFER_OUTPUT_DIR:?}" "${VIRTUAL_ENV:?}"
screen_root=$INFER_OUTPUT_DIR
screen_libs=$VIRTUAL_ENV/lib
for screen_lib in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib "$VIRTUAL_ENV"/lib/python*/site-packages/torch/lib "$VIRTUAL_ENV"/lib/python*/site-packages/triton/backends/nvidia/lib; do
  [[ ! -d "$screen_lib" ]] || screen_libs="$screen_libs:$screen_lib"
done
export LD_LIBRARY_PATH="$screen_libs:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$INFER_WEIGHTS_DIR/python-overlay:$INFER_CODE_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LLV2_USE_FA3=0 LLV2_USE_FA4=0 LLV2_USE_TE_ATTN=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$INFER_CODE_DIR"
if [[ ${1:-} == --prepare-only ]]; then
  python scripts/prepare_longlive2_reference.py --inputs "$INFER_WEIGHTS_DIR" --source "$INFER_CODE_DIR/third_party/LongLive2"
  touch "$screen_root/prepared.ok";exit 0
fi
IFS=',' read -r -a screen_gpus <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ ${#screen_gpus[@]} == 4 ]] || { echo 'requires four assigned GPUs' >&2; exit 2; }
screen_scenarios=(generated_patchwork_toy_cut_revisit generated_patchwork_toy_cut_revisit generated_bead_state_cut_revisit generated_bead_state_cut_revisit)
screen_seeds=(20260913 20260914 20260913 20260914)
screen_pids=()
for screen_lane in "${!screen_gpus[@]}"; do
  CUDA_VISIBLE_DEVICES=${screen_gpus[$screen_lane]} python scripts/run_longlive2_native_reference.py \
    --assets "$INFER_WEIGHTS_DIR" --output "$screen_root/lane$screen_lane" --seed "${screen_seeds[$screen_lane]}" \
    --cut-scenario "${screen_scenarios[$screen_lane]}" >"$screen_root/lane$screen_lane.log" 2>&1 &
  screen_pids+=("$!")
done
screen_statuses=()
for screen_pid in "${screen_pids[@]}"; do
  screen_status=0
  wait "$screen_pid" || screen_status=$?
  screen_statuses+=("$screen_status")
done
python scripts/collect_native_cut_screen.py --root "$screen_root" --exit-codes "${screen_statuses[@]}"
