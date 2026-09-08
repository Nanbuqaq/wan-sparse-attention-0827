#!/usr/bin/env bash
set -Eeuo pipefail
: "${INFER_CODE_DIR:?}" "${INFER_WEIGHTS_DIR:?}" "${INFER_OUTPUT_DIR:?}" "${VIRTUAL_ENV:?}"
study_libs=$VIRTUAL_ENV/lib
for study_lib in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib "$VIRTUAL_ENV"/lib/python*/site-packages/torch/lib "$VIRTUAL_ENV"/lib/python*/site-packages/triton/backends/nvidia/lib; do
  [[ ! -d "$study_lib" ]] || study_libs="$study_libs:$study_lib"
done
export LD_LIBRARY_PATH="$study_libs:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$INFER_WEIGHTS_DIR/python-overlay:$INFER_CODE_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LLV2_USE_FA3=0 LLV2_USE_FA4=0 LLV2_USE_TE_ATTN=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$INFER_CODE_DIR"
if [[ ${1:-} == --prepare-only ]]; then
  python scripts/prepare_longlive2_reference.py --inputs "$INFER_WEIGHTS_DIR" --source "$INFER_CODE_DIR/third_party/LongLive2"
  python -c 'from adapters.longlive_sparse.native_episode_memory import NativeEpisodeMemory; from scripts.collect_native_memory_study import validate_group; print("MEMORY_STUDY_IMPORT_PASS")'
  touch "$INFER_OUTPUT_DIR/prepared.ok"
  exit 0
fi
IFS=',' read -r -a study_gpus <<<"${CUDA_VISIBLE_DEVICES:?}"
[[ ${#study_gpus[@]} == 4 ]] || { echo 'requires four assigned GPUs' >&2; exit 2; }
: "${STUDY_ARMS:?runtime-frozen arm list required}" "${STUDY_SEEDS:?two comma-separated seeds required}" "${STUDY_SCENARIOS:?two comma-separated scenarios required}"
IFS=',' read -r -a study_seeds <<<"$STUDY_SEEDS"
IFS=',' read -r -a study_scenarios <<<"$STUDY_SCENARIOS"
[[ ${#study_seeds[@]} == 2 && ${#study_scenarios[@]} == 2 ]]
study_pids=()
for study_lane in "${!study_gpus[@]}"; do
  study_order=$STUDY_ARMS
  if (( study_lane % 2 )); then
    study_order=$(python -c 'import sys; print(",".join(reversed(sys.argv[1].split(","))))' "$STUDY_ARMS")
  fi
  CUDA_VISIBLE_DEVICES=${study_gpus[$study_lane]} python scripts/run_native_memory_study.py \
    --assets "$INFER_WEIGHTS_DIR" --output "$INFER_OUTPUT_DIR/lane$study_lane" \
    --cut-scenario "${study_scenarios[$((study_lane/2))]}" --seed "${study_seeds[$((study_lane%2))]}" \
    --arms "$study_order" >"$INFER_OUTPUT_DIR/lane$study_lane.log" 2>&1 &
  study_pids+=("$!")
done
for study_pid in "${study_pids[@]}"; do wait "$study_pid" || true; done
python scripts/collect_native_memory_study.py --root "$INFER_OUTPUT_DIR"
